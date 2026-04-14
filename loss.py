"""Loss computation for GiGPO with Future-KL (FIPO)."""

import torch
import torch.nn.functional as F


def compute_future_kl_chunked(
    log_ratio: torch.Tensor,
    dual_clip_threshold: float = 10.0,
    tau: float = 32.0,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Memory-efficient chunked Future-KL computation.
    
    FutureKL_t = Σ_{k=t}^{T} M_k · γ^{k-t} · Δ log p_k
    
    Uses chunked matrix multiplication to avoid O(L^2) memory.
    Time complexity: O(L^2), Memory: O(L * chunk_size)
    
    Args:
        log_ratio: Tensor of log probability ratios [seq_len]
        dual_clip_threshold: Threshold for filtering extreme IS ratios
        tau: Effective horizon (half-life) for decay
        chunk_size: Chunk size for memory efficiency
        
    Returns:
        Future-KL weights for each position [seq_len]
    """
    seq_len = log_ratio.shape[0]
    gamma = 2.0 ** (-1.0 / tau)
    
    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
    
    filter_threshold = torch.log(torch.tensor(dual_clip_threshold, device=log_ratio.device, dtype=log_ratio.dtype))
    participation_mask = log_ratio <= filter_threshold
    
    D = log_ratio * participation_mask.to(log_ratio.dtype)
    
    future_kl = torch.zeros_like(log_ratio)
    pos_i = torch.arange(seq_len, device=log_ratio.device).unsqueeze(1)
    
    gamma_t = torch.tensor(gamma, dtype=log_ratio.dtype, device=log_ratio.device)
    
    for j_start in range(0, seq_len, chunk_size):
        j_end = min(j_start + chunk_size, seq_len)
        j_idx = torch.arange(j_start, j_end, device=log_ratio.device).unsqueeze(0)
        
        distance = j_idx - pos_i
        mask = distance >= 0
        distance_clamped = distance.clamp(min=0)
        decay_block = torch.pow(gamma_t, distance_clamped) * mask.to(log_ratio.dtype)
        
        kl_block = D[j_start:j_end].unsqueeze(0)
        contrib = torch.matmul(kl_block, decay_block.t()).squeeze(0)
        future_kl += contrib
    
    return future_kl


def compute_loss_batch(
    model,
    ref_model,
    batch_transitions: list,
    batch_advantages: list,
    device: torch.device,
    beta: float = 0.02,
    pad_token_id: int = 0,
    use_future_kl: bool = False,
    future_kl_tau: float = 32.0,
    future_kl_clip_low: float = 1.0,
    future_kl_clip_high: float = 1.2,
    dual_clip_threshold: float = 10.0,
    future_kl_clip_high_only: bool = True,
    safety_thresh: float = 10.0,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
) -> torch.Tensor:
    """Compute GRPO loss for a batch of transitions with optional Future-KL (FIPO).
    
    Uses clipped surrogate objective for stability:
    L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL
    
    With Future-KL:
    Ã_t = Â_t · f_t, where f_t = clip(exp(FutureKL_t), low, high)
    
    Batch processing: process all transitions in parallel for efficiency.
    
    Args:
        pad_token_id: The token ID to use for padding (should match tokenizer.pad_token_id)
        use_future_kl: Whether to use Future-KL weighting
        future_kl_tau: Effective horizon (half-life) for decay
        future_kl_clip_low: Lower bound for influence weight clipping
        future_kl_clip_high: Upper bound for influence weight clipping
        dual_clip_threshold: Threshold for filtering extreme IS ratios in Future-KL
        future_kl_clip_high_only: If True, only clip upper bound (for larger models)
        safety_thresh: Safety threshold for negative samples with high IS ratio
        clip_ratio_low: Lower clip ratio for PPO
        clip_ratio_high: Upper clip ratio for PPO
    """
    if not batch_transitions:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    valid_transitions = []
    valid_advantages = []
    for t, adv in zip(batch_transitions, batch_advantages):
        if t.completion_ids:
            valid_transitions.append(t)
            valid_advantages.append(adv)
    
    if not valid_transitions:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    input_ids_list = []
    attention_mask_list = []
    prompt_lens = []
    completion_ids_list = []
    old_log_probs_list = []
    
    max_seq_len = 0
    for t in valid_transitions:
        seq_len = len(t.prompt_ids) + len(t.completion_ids)
        max_seq_len = max(max_seq_len, seq_len)
    
    for t in valid_transitions:
        if not t.prompt_ids:
            print(f"Warning: Empty prompt_ids in transition, skipping")
            continue
        input_ids = t.prompt_ids + t.completion_ids
        seq_len = len(input_ids)
        padding_len = max_seq_len - seq_len
        
        padded_input_ids = [pad_token_id] * padding_len + input_ids
        attention_mask = [0] * padding_len + [1] * seq_len
        
        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(attention_mask)
        prompt_lens.append(len(t.prompt_ids))
        completion_ids_list.append(t.completion_ids)
        old_log_probs_list.append(t.log_probs if t.log_probs else [])
    
    if not input_ids_list:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long, device=device)
    
    with torch.no_grad():
        ref_outputs = ref_model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )
        ref_logits = ref_outputs.logits
    
    outputs = model(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor,
    )
    logits = outputs.logits
    
    losses = []
    
    for i, (t, advantage) in enumerate(zip(valid_transitions, valid_advantages)):
        prompt_len = prompt_lens[i]
        completion_ids = completion_ids_list[i]
        old_log_probs = old_log_probs_list[i]
        
        completion_len = len(completion_ids)
        
        start_idx = max_seq_len - prompt_len - completion_len
        end_idx = max_seq_len - prompt_len
        
        if start_idx < 0:
            start_idx = 0
        
        curr_logits = logits[i, start_idx:end_idx, :]
        curr_ref_logits = ref_logits[i, start_idx:end_idx, :]
        
        if curr_logits.shape[0] != completion_len:
            curr_logits = curr_logits[-completion_len:, :]
            curr_ref_logits = curr_ref_logits[-completion_len:, :]
        
        completion_tensor = torch.tensor(completion_ids, dtype=torch.long, device=device)
        
        new_log_probs = F.log_softmax(curr_logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze(1)
        
        ref_log_probs = F.log_softmax(curr_ref_logits, dim=-1)
        ref_log_probs = ref_log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze(1)
        
        if old_log_probs and len(old_log_probs) == completion_len:
            old_log_probs_tensor = torch.tensor(old_log_probs, dtype=new_log_probs.dtype, device=device)
        else:
            old_log_probs_tensor = new_log_probs.detach().clone()
        
        log_ratio = new_log_probs - old_log_probs_tensor
        ratio = torch.exp(log_ratio)
        
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
        
        advantage_tensor = torch.tensor(advantage, dtype=new_log_probs.dtype, device=device)
        
        if use_future_kl and completion_len > 1:
            future_kl = compute_future_kl_chunked(
                log_ratio=log_ratio,
                dual_clip_threshold=dual_clip_threshold,
                tau=future_kl_tau,
            )
            
            influence_weight = torch.exp(future_kl)
            
            if future_kl_clip_high_only:
                influence_weight = torch.clamp(influence_weight, min=1.0, max=future_kl_clip_high)
            else:
                influence_weight = torch.clamp(
                    influence_weight,
                    min=future_kl_clip_low,
                    max=future_kl_clip_high,
                )
            
            if advantage_tensor.item() < 0:
                mask_neg_high_is = ratio > safety_thresh
                if mask_neg_high_is.any():
                    influence_weight = torch.where(
                        mask_neg_high_is,
                        torch.clamp(influence_weight, min=0.8, max=1.0),
                        influence_weight
                    )
            
            modified_advantage = advantage_tensor * influence_weight
        else:
            modified_advantage = advantage_tensor.expand(completion_len).clone()
        
        surr1 = ratio * modified_advantage
        surr2 = clipped_ratio * modified_advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        
        kl_div = ref_log_probs - new_log_probs
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        kl_loss = per_token_kl.mean()
        
        loss = policy_loss - beta * kl_loss
        losses.append(loss)
    
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    batch_loss = torch.stack(losses).mean()
    return batch_loss
