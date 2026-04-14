loss_mode=future_kl
decay_rate=32.0
chunk_size=128
future_kl_start=include_current
future_kl_window=-1
future_kl_average=False
future_kl_clip_ratio=0.2
future_kl_clip_high_only=True
safety_thresh=10.0

clip_ratio_low=0.2
clip_ratio_high=0.28

def compute_policy_loss_future_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # let's compute the future kl, which is the kl accumulated from the current token to the end of the response(within the response_mask)

    assert log_prob.shape == old_log_prob.shape == advantages.shape, \
        f"log/old/adv shape mismatch: {log_prob.shape}, {old_log_prob.shape}, {advantages.shape}"
    
    assert response_mask.dim() == 2 and response_mask.size(0) == log_prob.size(0), \
        f"response_mask shape {response_mask.shape} incompatible with batch {log_prob.shape}"

    # calculate future_kl using negative_approx_kl and response_mask
    batch_size, response_len = log_prob.shape
    device = log_prob.device

    assert response_mask.size(1) == response_len, \
        f"Time dim mismatch: log_prob length={response_len}, response_mask length={response_mask.size(1)}"

    chunk_size = config.policy_loss.get('chunk_size', 128)
    decay_rate = config.policy_loss.get('decay_rate',128)
    gamma = 2 ** (-1.0 / decay_rate) 

    future_kl = torch.zeros((batch_size, response_len), device=device, dtype=log_prob.dtype)
    pos_i = torch.arange(response_len, device=device).unsqueeze(1)  # (L,1)
    # to avoid high is token from the sample to deviate our training and weighting 
    # we exclude those greater than clip_frac_c. These tokens have no gradient neither in the following training. 
    filter_threshold = torch.log(torch.tensor(clip_ratio_c, device=device, dtype=log_prob.dtype))    
    is_negative_adv = (advantages < 0)#.view(batch_size, 1) # bsz, L
    ignore_mask = negative_approx_kl > filter_threshold # bsz, L
    participation_mask = ~ignore_mask
    kl_response_premask = negative_approx_kl * response_mask.to(log_prob.dtype) # response mased kl diff
    kl_response = kl_response_premask * participation_mask.to(log_prob.dtype) 
    
    gamma_t = torch.tensor(gamma, dtype=log_prob.dtype, device=device)
    for j_start in range(0, response_len, chunk_size):
        j_end = min(response_len, j_start + chunk_size)
        j_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)  # (1, Kb)
        # distance shape (L, Kb) where entry (i,k) = j - i
        distance = j_idx - pos_i
        mask = distance >= 0  # zero out j < i
        distance_clamped = distance.clamp(min=0)
        # decay_block (L, Kb)
        decay_block = torch.pow(gamma_t, distance_clamped) * mask.to(log_prob.dtype)
        # kl_block (B, Kb)
        kl_block = kl_response[:, j_start:j_end]
        # contribution: for this block, contrib_{b,i} = sum_k kl_block[b,k] * decay_block[i,k]
        # compute via matmul: (B, Kb) @ (Kb, L) -> (B, L)
        contrib = torch.matmul(kl_block, decay_block.t())
        future_kl += contrib

    if config.policy_loss.get("future_kl_clip_ratio") != 0.0:
        clip_ratio = config.policy_loss.get("future_kl_clip_ratio")
        if not config.policy_loss.get('future_kl_clip_high_only'):
            # seems to work well with smaller models such as 7b --> usually create lower entropy
            upper_bound = 1.0 + clip_ratio
            lower_bound = 1.0 - clip_ratio
            influence_weights = torch.clamp(torch.exp(future_kl),  min=lower_bound,max=upper_bound).detach()
        else:
            # a radical way to update model, works fine for larger model to break boundary hopefully
            upper_bound = 1.0 + clip_ratio
            lower_bound = 1.0
            influence_weights = torch.clamp(torch.exp(future_kl),  min=1.0,max=1.0+clip_ratio).detach()
    else:
        upper_bound = 10.0
        lower_bound = 0.0
        influence_weights = torch.clamp(torch.exp(future_kl),  max=10.0).detach()
    # Apply a safety threshold: if a negative sample's IS value is too high and its weight is increasing, cap it at the baseline value (1.0)
    # To avoid over-penalization
    safe_threshold = config.policy_loss.get('safety_thresh', 4.0)
    mask_neg_high_is = (advantages < 0) & (ratio > safe_threshold)
    influence_weights = torch.where(mask_neg_high_is, torch.clamp(influence_weights, min=0.8, max=1.0), influence_weights)
    # calcuate clip ratio
    clip_frac_upper = verl_F.masked_mean((influence_weights >= upper_bound - 1e-7).float(), response_mask)
    clip_frac_lower = verl_F.masked_mean((influence_weights <= lower_bound + 1e-7).float(), response_mask)
    total_clip_frac = clip_frac_upper + clip_frac_lower
    # add stats for raw influence weight 
    influence_weights_mean_raw = verl_F.masked_mean(torch.exp(future_kl), response_mask)
    valid_vals_raw = torch.exp(future_kl)[response_mask.to(dtype=torch.bool, device=influence_weights.device)]
    raw_influence_weights_min = valid_vals_raw.min()
    raw_influence_weights_max = valid_vals_raw.max()
    # add status check of the influence_weights
    influence_weights_mean = verl_F.masked_mean(influence_weights, response_mask)
    # influence_weights_std = verl_F.masked_std(influence_weights, response_mask)
    valid_vals = influence_weights[response_mask.to(dtype=torch.bool, device=influence_weights.device)]
    influence_weights_min = valid_vals.min()
    influence_weights_max = valid_vals.max()
    
    weighted_advantages = advantages * influence_weights

    pg_losses1 = -weighted_advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -weighted_advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    # Stats info to collect：
    # raw influence weight，lower clip token count, done
    # Percentages of IS negative samples > 2, 3, 4, 10 
    # Negative sample is: max，995 percent，999 percent, done
    # Positive sample is: max，995 percent，999 percent, done
    # Extremely small is from Positive sample (could result in large distribution shift)

    # filter mechanism： if a sequence contains more than 1 token that has been clip by dual clip, then we throw away the entire sequence. 
    # however, this is rarely activated in the 32b training. 
    lower_clip_mask = (
                        (advantages < 0) &
                        (clip_pg_losses1 > pg_losses3) &
                        response_mask.bool()
                    ) 
    low_clip_token_counts = lower_clip_mask.sum(dim=1)  # (batch,）

    # sequence-level: whether this entire response should be invalidated
    seq_has_low_clip = (low_clip_token_counts > 1)        # (batch,) # hard threshold (if sequence has many, > threshold,--> sequence)
    seq_valid_mask = (~seq_has_low_clip).unsqueeze(1)   # (batch,1)

    final_mask = response_mask.bool() & seq_valid_mask  # (batch, response_len)
    final_mask_f = final_mask.to(log_prob.dtype)

    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=final_mask_f, loss_agg_mode=loss_agg_mode)

    # Start gathering the rest of stats information
    neg_ratio_2_3 = verl_F.masked_mean(((ratio >= 2.0) & (ratio < 3.0) & is_negative_adv).float(), response_mask)
    neg_ratio_3_4 = verl_F.masked_mean(((ratio >= 3.0) & (ratio < 4.0) & is_negative_adv).float(), response_mask)
    neg_ratio_4_10 = verl_F.masked_mean(((ratio >= 4.0) & (ratio < clip_ratio_c) & is_negative_adv).float(), response_mask)
    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    if neg_valid.numel() > 0:
        neg_is_max = neg_valid.max()
        neg_is_p75 = torch.quantile(neg_valid, 0.75)
        neg_is_p995 = torch.quantile(neg_valid, 0.995)
        neg_is_p999 = torch.quantile(neg_valid, 0.999)
    else:
        neg_is_max = torch.tensor(0.0, device=ratio.device)
        neg_is_p995 = torch.tensor(0.0, device=ratio.device)
        neg_is_p999 = torch.tensor(0.0, device=ratio.device)
        neg_is_p75 = torch.tensor(0.0, device=ratio.device)

    pos_valid = ratio[(advantages > 0) & response_mask.bool()]
    if pos_valid.numel() > 0:
        pos_is_max = pos_valid.max()
        pos_is_p25 = torch.quantile(pos_valid, 0.25)
        pos_is_median = torch.quantile(pos_valid, 0.5)
        pos_is_p75 = torch.quantile(pos_valid, 0.75)
        pos_is_p995 = torch.quantile(pos_valid, 0.995)
        pos_is_p999 = torch.quantile(pos_valid, 0.999)
        pos_is_min = pos_valid.min()
    else:
        pos_is_p25 = torch.tensor(0.0, device=ratio.device)
        pos_is_max = torch.tensor(0.0, device=ratio.device)
        pos_is_median = torch.tensor(0.0, device=ratio.device)
        pos_is_p75 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p999 = torch.tensor(0.0, device=ratio.device)
        pos_is_min = torch.tensor(0.0, device=ratio.device)

    pos_mini_frac = verl_F.masked_mean(((ratio < 1e-3) & (advantages > 0)).float(), response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, influence_weights_mean, influence_weights_min, influence_weights_max, total_clip_frac, clip_frac_upper, clip_frac_lower, \
           influence_weights_mean_raw, raw_influence_weights_min, raw_influence_weights_max,neg_ratio_2_3, neg_ratio_3_4, neg_ratio_4_10, \
           neg_is_max, neg_is_p995,neg_is_p999,neg_is_p75, pos_is_max, pos_is_median, pos_is_p75, pos_is_p995, pos_is_p999,pos_is_p25, pos_is_min, pos_mini_frac, kl_response_premask