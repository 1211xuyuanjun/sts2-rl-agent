"""Main entry point for STS2 GRPO training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gigpo.config import STS2GRPOConfig
from gigpo.trainer import STS2GRPOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="STS2 GRPO Training")
    
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer-id", type=str, default=None, help="Tokenizer ID (defaults to model_id)")
    parser.add_argument("--dataset-size", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, default="outputs/sts2_grpo")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--player-hp", type=int, default=80, help="Player starting HP")
    parser.add_argument("--player-max-hp", type=int, default=80, help="Player max HP")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for GiGPO")
    parser.add_argument("--omega", type=float, default=1.0, help="Weight for step-level advantage in GiGPO")
    parser.add_argument("--use-std-norm", action="store_true", default=True, help="Use std normalization for advantages")
    parser.add_argument("--format-reward-weight", type=float, default=1.0, help="Weight for format reward")
    parser.add_argument("--use-vllm", action="store_true", default=False, help="Use vLLM for rollout")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA for training")
    parser.add_argument("--no-lora", action="store_false", dest="use_lora", help="Disable LoRA (full finetuning)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use-gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    
    # Future-KL (FIPO) parameters
    parser.add_argument("--use-future-kl", action="store_true", default=False, help="Use Future-KL (FIPO)")
    parser.add_argument("--future-kl-tau", type=float, default=32.0, help="Future-KL tau (half-life)")
    parser.add_argument("--future-kl-clip-low", type=float, default=1.0, help="Future-KL clip lower bound")
    parser.add_argument("--future-kl-clip-high", type=float, default=1.2, help="Future-KL clip upper bound")
    parser.add_argument("--dual-clip-threshold", type=float, default=10.0, help="Dual-clip threshold for Future-KL")
    parser.add_argument("--future-kl-clip-high-only", action="store_true", default=True, help="Only clip upper bound (for larger models)")
    parser.add_argument("--safety-thresh", type=float, default=10.0, help="Safety threshold for negative samples")
    parser.add_argument("--clip-ratio-low", type=float, default=0.2, help="PPO clip ratio lower bound")
    parser.add_argument("--clip-ratio-high", type=float, default=0.28, help="PPO clip ratio upper bound")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for LR scheduler")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Min LR ratio for cosine scheduler")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    config = STS2GRPOConfig(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        dataset_size=args.dataset_size,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        beta=args.beta,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        player_hp=args.player_hp,
        player_max_hp=args.player_max_hp,
        bf16=args.bf16,
        debug=args.debug,
        gamma=args.gamma,
        omega=args.omega,
        use_std_norm=args.use_std_norm,
        format_reward_weight=args.format_reward_weight,
        use_vllm=args.use_vllm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_future_kl=args.use_future_kl,
        future_kl_tau=args.future_kl_tau,
        future_kl_clip_low=args.future_kl_clip_low,
        future_kl_clip_high=args.future_kl_clip_high,
        dual_clip_threshold=args.dual_clip_threshold,
        future_kl_clip_high_only=args.future_kl_clip_high_only,
        safety_thresh=args.safety_thresh,
        clip_ratio_low=args.clip_ratio_low,
        clip_ratio_high=args.clip_ratio_high,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    
    trainer = STS2GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
