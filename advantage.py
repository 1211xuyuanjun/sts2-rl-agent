"""Advantage computation for GiGPO."""

import torch


def compute_trajectory_advantage(transitions: list) -> None:
    """Compute trajectory-level relative advantage.
    
    Group by seed_idx first, then within each seed group, compute advantage
    based on final_reward across trajectories.
    This ensures trajectories from different seeds are compared separately.
    """
    seed_groups = {}
    for t in transitions:
        seed_idx = t.seed_idx
        if seed_idx not in seed_groups:
            seed_groups[seed_idx] = []
        seed_groups[seed_idx].append(t)
    
    for seed_idx, seed_transitions in seed_groups.items():
        trajectory_groups = {}
        for t in seed_transitions:
            traj_id = t.trajectory_id
            if traj_id not in trajectory_groups:
                trajectory_groups[traj_id] = []
            trajectory_groups[traj_id].append(t)
        
        trajectory_final_rewards = {}
        for traj_id, traj_transitions in trajectory_groups.items():
            if traj_transitions:
                trajectory_final_rewards[traj_id] = traj_transitions[0].final_reward
        
        if len(trajectory_final_rewards) > 1:
            final_rewards = torch.tensor(list(trajectory_final_rewards.values()))
            mean_r = final_rewards.mean()
            std_r = final_rewards.std() + 1e-8
            normalized_advantages = (final_rewards - mean_r) / std_r
            
            traj_ids = list(trajectory_final_rewards.keys())
            for traj_id, adv in zip(traj_ids, normalized_advantages):
                for t in trajectory_groups[traj_id]:
                    t.trajectory_relative_advantage = adv.item()
        else:
            for t in seed_transitions:
                t.trajectory_relative_advantage = 0.0


def compute_relative_advantage(transitions: list, gamma: float, omega: float, use_std_norm: bool = True) -> list:
    """Compute relative advantage for each transition using GiGPO.
    
    GiGPO (Group-in-Group Policy Optimization) uses two-level advantages:
    1. Episode-level relative advantages (A^E): based on final_reward across trajectories
    2. Step-level relative advantages (A^S): based on discounted returns within anchor state groups
    
    The step-level advantage uses discounted return:
    R_t^{(i)} = sum_{k=t}^{T} gamma^{k-t} * r_k^{(i)}
    
    Total advantage: A = A^E + omega * A^S
    
    Returns: list of (transition, total_advantage) tuples
    """
    compute_trajectory_advantage(transitions)
    
    trajectory_groups = {}
    for t in transitions:
        traj_key = (t.seed_idx, t.trajectory_id)
        if traj_key not in trajectory_groups:
            trajectory_groups[traj_key] = []
        trajectory_groups[traj_key].append(t)
    
    for traj_key, traj_transitions in trajectory_groups.items():
        traj_transitions.sort(key=lambda x: x.state.get("step_num", 0) if isinstance(x.state, dict) else 0)
        
        discounted_returns = []
        cumulative_reward = 0.0
        for t in reversed(traj_transitions):
            cumulative_reward = t.step_reward + t.format_reward + gamma * cumulative_reward
            discounted_returns.append(cumulative_reward)
        discounted_returns.reverse()
        
        for t, ret in zip(traj_transitions, discounted_returns):
            t.discounted_return = ret
    
    anchor_state_groups = {}
    for t in transitions:
        anchor_state = t.state_text
        if anchor_state not in anchor_state_groups:
            anchor_state_groups[anchor_state] = []
        anchor_state_groups[anchor_state].append(t)
    
    for anchor_state, group in anchor_state_groups.items():
        returns = torch.tensor([t.discounted_return for t in group])
        
        if len(group) > 1:
            if use_std_norm:
                mean_r = returns.mean()
                std_r = returns.std() + 1e-8
                advantages = (returns - mean_r) / std_r
            else:
                mean_r = returns.mean()
                advantages = returns - mean_r
        else:
            advantages = torch.zeros(1)
        
        for t, adv in zip(group, advantages):
            t.step_relative_advantage = adv.item()
    
    results = []
    for t in transitions:
        total_advantage = t.trajectory_relative_advantage + omega * t.step_relative_advantage
        results.append((t, total_advantage))
    
    return results
