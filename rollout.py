"""Rollout collection for GiGPO."""

import copy
import gc
import os
from typing import Optional

import torch
from tqdm import tqdm

from .prompts import SYSTEM_PROMPT_BASE, make_state_prompt, parse_action, check_format_reward
from .transition import Transition


class RolloutCollector:
    """Collects rollouts for GiGPO training."""
    
    def __init__(self, env, config, device, dtype, tokenizer, lora_adapter_path=None):
        self.tokenizer = tokenizer
        self.env = env
        self.config = config
        self.device = device
        self.dtype = dtype
        
        self.vllm_available = False
        self.vllm_llm = None
        self.lora_adapter_path = lora_adapter_path
        self.lora_request = None
    
    def load_vllm(self):
        """Load vLLM engine for rollout phase.
        
        Call this before Phase 1 (rollout) to enable fast inference.
        """
        if self.config.use_vllm and not self.vllm_available:
            try:
                from vllm import LLM
                from vllm.lora.request import LoRARequest
                
                if self.config.use_lora:
                    if self.lora_adapter_path is None:
                        raise ValueError("LoRA adapter path not set. Call save_lora_adapter first.")
                    
                    import os
                    if not os.path.exists(self.lora_adapter_path):
                        raise ValueError(
                            f"LoRA adapter path does not exist: {self.lora_adapter_path}\n"
                            "Make sure to save the adapter before loading vLLM."
                        )
                    
                    print(f"Loading vLLM with LoRA support...")
                    print(f"  Model: {self.config.model_id}")
                    print(f"  LoRA adapter: {self.lora_adapter_path}")
                    print(f"  Max LoRA rank: {self.config.lora_r}")
                    
                    self.vllm_llm = LLM(
                        model=self.config.model_id,
                        dtype="bfloat16" if self.config.bf16 else "float16",
                        max_model_len=self.config.max_seq_len,
                        enable_lora=True,
                        max_loras=1,
                        max_lora_rank=self.config.lora_r,
                        disable_log_stats=True,
                    )

                    self.lora_request = LoRARequest(
                        lora_name="current_adapter",
                        lora_int_id=1,
                        lora_local_path=self.lora_adapter_path,
                    )
                    print(f"vLLM loaded with LoRA support from: {self.lora_adapter_path}")
                else:
                    print(f"Loading vLLM for rollout...")
                    print(f"  Model: {self.config.model_id}")
                    
                    self.vllm_llm = LLM(
                        model=self.config.model_id,
                        dtype="bfloat16" if self.config.bf16 else "float16",
                        max_model_len=self.config.max_seq_len,
                        gpu_memory_utilization=0.9,
                        disable_log_stats=True,
                    )
                    print("vLLM loaded for rollout")
                self.vllm_available = True
            except ImportError as e:
                raise ImportError(
                    "vLLM is required for rollout. Please install vLLM: pip install vllm\n"
                    f"Error: {e}"
                )
            except Exception as e:
                print(f"Error loading vLLM: {e}")
                print(f"  Model ID: {self.config.model_id}")
                print(f"  LoRA adapter path: {self.lora_adapter_path}")
                print(f"  GPU available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                raise
    
    def unload_vllm(self):
        """Unload vLLM engine to free memory.
        
        Call this after Phase 1 (rollout) before Phase 2 (advantage computation).
        """
        if self.vllm_available and self.vllm_llm is not None:
            # 先清理 LoRA request
            if self.lora_request is not None:
                del self.lora_request
                self.lora_request = None
            
            # 删除 vLLM 引擎
            del self.vllm_llm
            self.vllm_llm = None
            self.vllm_available = False
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("vLLM unloaded to free memory")
    
    def generate_action_vllm(self, state_text: str) -> tuple[int, list[int], list[int], list[float]]:
        """Generate action using vLLM.
        
        Args:
            state_text: State description (user message content)
        
        Returns: (action, prompt_ids, completion_ids, log_probs)
        """
        from vllm import SamplingParams
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": state_text},
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            max_tokens=self.config.max_new_tokens,
            logprobs=1,
        )
        
        lora_request = self.lora_request if self.config.use_lora else None
        outputs = self.vllm_llm.generate([prompt_text], sampling_params, lora_request=lora_request, use_tqdm=False)
        output = outputs[0]
        
        completion_ids = output.outputs[0].token_ids
        completion_text = output.outputs[0].text
        action = parse_action(completion_text)
        
        log_probs = []
        if output.outputs[0].logprobs:
            for i, token_logprob in enumerate(output.outputs[0].logprobs):
                if token_logprob and i < len(completion_ids):
                    token_id = completion_ids[i]
                    if token_id in token_logprob:
                        log_probs.append(token_logprob[token_id].logprob)
        
        return action, prompt_ids, list(completion_ids), log_probs
    
    def generate_action(self, state_text: str) -> tuple[int, list[int], list[int], list[float]]:
        """Generate action from prompt using vLLM.
        
        Args:
            state_text: State description (user message content)
        
        Returns: (action, prompt_ids, completion_ids, log_probs)
        """
        if not self.vllm_available or self.vllm_llm is None:
            raise RuntimeError("vLLM not loaded. Call load_vllm() first.")
        
        return self.generate_action_vllm(state_text)
    
    def rollout_once(self, seed_prompt: str, trajectory_id: int, seed_idx: int, pbar: Optional[tqdm] = None, seed: Optional[int] = None) -> list:
        """Run one rollout and collect transitions.
        
        Args:
            seed_prompt: The seed prompt (goal)
            trajectory_id: ID for this trajectory
            seed_idx: Index of the seed prompt
            pbar: Progress bar to update
            seed: Random seed for env.reset() to ensure reproducible initial states
            
        Returns:
            List of Transition objects
        """
        obs, _ = self.env.reset(seed=seed)
        transitions = []
        step_num = 0
        
        while step_num < self.config.max_steps:
            state_text = make_state_prompt(obs)
            
            action, prompt_ids, completion_ids, log_probs = self.generate_action(state_text)
            
            next_obs, step_reward, terminated, truncated, info = self.env.step(action)
            
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            raw_format_reward = check_format_reward(completion_text)
            format_reward = raw_format_reward * self.config.format_reward_weight
            
            transition = Transition(
                state=copy.deepcopy(obs),
                state_text=state_text,
                action=action,
                step_reward=step_reward,
                final_reward=0.0,
                format_reward=format_reward,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                log_probs=log_probs,
                trajectory_id=trajectory_id,
                seed_idx=seed_idx,
            )
            transitions.append(transition)
            
            if pbar is not None:
                pbar.update(1)
            
            obs = next_obs
            step_num += 1
            
            if terminated or truncated:
                break
        
        if transitions:
            hp_str = obs.get("hp", "0/0")
            try:
                hp_parts = hp_str.split("/")
                current_hp = int(hp_parts[0])
                max_hp = int(hp_parts[1]) if len(hp_parts) > 1 else self.config.player_max_hp
            except:
                current_hp = 0
                max_hp = self.config.player_max_hp
            
            hp_ratio = current_hp / max_hp if max_hp > 0 else 0
            
            if current_hp > 0:
                final_reward = hp_ratio
            else:
                final_reward = -1.0
            
            for t in transitions:
                t.final_reward = final_reward
                t.total_reward = t.step_reward + t.final_reward + t.format_reward
        
        return transitions
    
    def collect_rollout_batch(self, seed_prompt: str, trajectory_id: int, seed_idx: int) -> list:
        """Collect rollout for one trajectory.
        
        Returns transitions for one trajectory.
        """
        with tqdm(total=self.config.max_steps, desc=f"Rollout", unit="step", leave=False) as pbar:
            transitions = self.rollout_once(seed_prompt, trajectory_id, seed_idx, pbar)
        return transitions
    
    def collect_rollouts(self, seed_prompts: list, seeds: list = None) -> list:
        """Collect rollouts for all seeds.
        
        For each seed, generate N trajectories.
        
        Args:
            seed_prompts: List of seed prompts (goals)
            seeds: List of random seeds for env.reset() (one per seed_prompt)
        """
        all_transitions = []
        total_rollouts = len(seed_prompts) * self.config.num_generations
        
        if seeds is None:
            seeds = [None] * len(seed_prompts)
        
        with tqdm(total=total_rollouts * self.config.max_steps, desc="Rollout", unit="step") as pbar:
            for seed_idx, (seed_prompt, seed) in enumerate(zip(seed_prompts, seeds)):
                for traj_idx in range(self.config.num_generations):
                    if self.config.debug:
                        print(f"Rollout: seed {seed_idx+1}/{len(seed_prompts)}, traj {traj_idx+1}/{self.config.num_generations}")
                    
                    transitions = self.rollout_once(seed_prompt, traj_idx, seed_idx, pbar, seed=seed)
                    all_transitions.extend(transitions)
        
        return all_transitions
