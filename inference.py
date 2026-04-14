"""Inference script for STS2 GRPO trained model using vLLM."""

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.gigpo.prompts import SYSTEM_PROMPT_BASE, make_state_prompt, parse_action


class STS2Inference:
    """Inference engine for STS2 using vLLM."""
    
    def __init__(self, model_id: str, adapter_dir: str = None, use_lora: bool = True):
        """Initialize inference engine.
        
        Args:
            model_id: Base model ID
            adapter_dir: Path to LoRA adapter directory
            use_lora: Whether to use LoRA adapter
        """
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.use_lora = use_lora
        
        self.vllm_llm = None
        self.tokenizer = None
        self.lora_request = None
        
        self._load_model()
    
    def _load_model(self):
        """Load vLLM model and tokenizer."""
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer
        
        print(f"Loading model with vLLM: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        if self.use_lora and self.adapter_dir:
            adapter_path = Path(self.adapter_dir)
            
            if (adapter_path / "adapter_config.json").exists():
                print(f"Loading LoRA adapter from: {self.adapter_dir}")
                
                self.vllm_llm = LLM(
                    model=self.model_id,
                    max_model_len=8192,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    enable_lora=True,
                    max_loras=1,
                    max_lora_rank=16,
                    disable_log_stats=True,
                )
                
                self.lora_request = LoRARequest(
                    lora_name="current_adapter",
                    lora_int_id=1,
                    lora_local_path=self.adapter_dir,
                )
                
                print(f"vLLM loaded with LoRA support from: {self.adapter_dir}")
            else:
                raise FileNotFoundError(
                    f"No adapter_config.json found in {self.adapter_dir}\n"
                    "Please ensure the adapter directory contains valid LoRA files."
                )
        else:
            print("Loading vLLM without LoRA...")
            
            self.vllm_llm = LLM(
                model=self.model_id,
                max_model_len=8192,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
                disable_log_stats=True,
            )
            
            print("vLLM loaded for inference")
    
    def generate_action(self, state_text: str, temperature: float = 0.7, top_k: int = 20, max_tokens: int = 1024):
        """Generate action from state text.
        
        Args:
            state_text: State description
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            action: Parsed action integer
            completion_text: Generated text
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
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        
        if self.lora_request is not None:
            outputs = self.vllm_llm.generate(
                [prompt_text],
                sampling_params,
                lora_request=self.lora_request,
                use_tqdm=False,
            )
        else:
            outputs = self.vllm_llm.generate(
                [prompt_text],
                sampling_params,
                use_tqdm=False,
            )
        
        completion_text = outputs[0].outputs[0].text
        action = parse_action(completion_text)
        
        return action, completion_text
    
    def run_episode(self, env, max_steps: int = 100, seed: int = None, verbose: bool = True, player_max_hp: int = 80):
        """Run one episode of inference.
        
        Args:
            env: STS2 environment
            max_steps: Maximum steps per episode
            seed: Random seed for environment
            verbose: Whether to print step details
            player_max_hp: Player's maximum HP for final reward calculation
            
        Returns:
            total_reward: Total step reward accumulated
            final_reward: Final reward based on HP
            step_num: Number of steps taken
        """
        obs, _ = env.reset(seed=seed)
        
        if verbose:
            print("\n" + "=" * 80)
            print("Starting Inference Episode")
            print("=" * 80)
        
        total_reward = 0.0
        step_num = 0
        
        while step_num < max_steps:
            state_text = make_state_prompt(obs)
            
            action, completion_text = self.generate_action(state_text)
            
            if action is None:
                if verbose:
                    print(f"  Warning: Failed to parse action, using ACTION_END_TURN (0)")
                action = 0
            
            if verbose:
                print(f"\nStep {step_num + 1}:")
                print(f"  Action: {action}")
            
            next_obs, step_reward, terminated, truncated, info = env.step(action)
            total_reward += step_reward
            
            obs = next_obs
            step_num += 1
            
            if terminated or truncated:
                if verbose:
                    print(f"\nEpisode ended at step {step_num}")
                break
        
        hp_str = obs.get("hp", "0/0")
        
        try:
            hp_parts = hp_str.split("/")
            current_hp = int(hp_parts[0])
            max_hp = int(hp_parts[1]) if len(hp_parts) > 1 else player_max_hp
        except:
            current_hp = 0
            max_hp = player_max_hp
        
        hp_ratio = current_hp / max_hp if max_hp > 0 else 0
        
        if current_hp > 0:
            final_reward = hp_ratio
        else:
            final_reward = -1.0
        
        if verbose:
            print(f"\nFinal HP: {hp_str}")
            print(f"Total Step Reward: {total_reward:.4f}")
            print(f"Final Reward: {final_reward:.4f}")
            print(f"Steps Taken: {step_num}")
        
        return total_reward, final_reward, step_num
    
    def cleanup(self):
        """Clean up resources."""
        if self.lora_request is not None:
            del self.lora_request
            self.lora_request = None
        
        if self.vllm_llm is not None:
            del self.vllm_llm
            self.vllm_llm = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Inference engine cleaned up")


def main():
    parser = argparse.ArgumentParser(description="STS2 GRPO Inference with vLLM")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", type=str, help="Path to LoRA adapter directory")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--player-hp", type=int, default=80)
    parser.add_argument("--player-max-hp", type=int, default=80)
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    from sts2_env.gym_env.combat_env import STS2CombatEnv
    
    env = STS2CombatEnv(
        player_hp=args.player_hp,
        player_max_hp=args.player_max_hp,
        max_turns=args.max_steps,
        use_dict_obs=True,
    )
    
    inference = STS2Inference(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        use_lora=args.use_lora,
    )
    
    try:
        episode_results = []
        
        for episode in range(args.num_episodes):
            print(f"\n{'='*80}")
            print(f"Episode {episode + 1}/{args.num_episodes} (seed={episode})")
            print(f"{'='*80}")
            
            total_reward, final_reward, steps = inference.run_episode(
                env=env,
                max_steps=args.max_steps,
                seed=episode,
                verbose=True,
                player_max_hp=args.player_max_hp,
            )
            
            episode_results.append({
                'episode': episode + 1,
                'seed': episode,
                'total_reward': total_reward,
                'final_reward': final_reward,
                'steps': steps,
            })
        
        print(f"\n{'='*80}")
        print("Episode Results Summary")
        print(f"{'='*80}")
        print(f"{'Episode':<10} {'Seed':<10} {'Total Reward':<15} {'Final Reward':<15} {'Steps':<10}")
        print("-" * 80)
        
        for result in episode_results:
            print(f"{result['episode']:<10} {result['seed']:<10} "
                  f"{result['total_reward']:<15.4f} {result['final_reward']:<15.4f} {result['steps']:<10}")
        
        if args.num_episodes > 1:
            total_rewards = [r['total_reward'] for r in episode_results]
            final_rewards = [r['final_reward'] for r in episode_results]
            steps_list = [r['steps'] for r in episode_results]
            
            print(f"\n{'='*80}")
            print("Statistics")
            print(f"{'='*80}")
            print(f"Total Reward  - Mean: {sum(total_rewards) / len(total_rewards):.4f}, "
                  f"Min: {min(total_rewards):.4f}, Max: {max(total_rewards):.4f}")
            print(f"Final Reward  - Mean: {sum(final_rewards) / len(final_rewards):.4f}, "
                  f"Min: {min(final_rewards):.4f}, Max: {max(final_rewards):.4f}")
            print(f"Steps         - Mean: {sum(steps_list) / len(steps_list):.1f}, "
                  f"Min: {min(steps_list)}, Max: {max(steps_list)}")
    finally:
        inference.cleanup()


if __name__ == "__main__":
    main()
