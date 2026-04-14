"""Transition data class for GiGPO."""


class Transition:
    """A single transition in a trajectory."""
    
    def __init__(
        self,
        state: dict,
        state_text: str,
        action: int,
        step_reward: float,
        final_reward: float,
        format_reward: float,
        prompt_ids: list[int],
        completion_ids: list[int],
        log_probs: list[float],
        trajectory_id: int = 0,
        seed_idx: int = 0,
    ):
        self.state = state
        self.state_text = state_text
        self.action = action
        self.step_reward = step_reward
        self.final_reward = final_reward
        self.format_reward = format_reward
        self.total_reward = step_reward + final_reward + format_reward
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids
        self.log_probs = log_probs
        self.trajectory_id = trajectory_id
        self.seed_idx = seed_idx
        self.trajectory_relative_advantage = 0.0
        self.step_relative_advantage = 0.0
        self.discounted_return = 0.0
