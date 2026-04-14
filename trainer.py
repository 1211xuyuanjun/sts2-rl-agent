"""GiGPO Trainer for STS2."""

import gc
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .advantage import compute_relative_advantage
from .config import STS2GRPOConfig
from .loss import compute_loss_batch
from .rollout import RolloutCollector
from .transition import Transition


class STS2GRPOTrainer:
    """GRPO Trainer for STS2.
    
    Algorithm:
    1. For each epoch, rollout ALL trajectories first
    2. Compute advantages for ALL transitions
    3. Create dataloader from transitions, compute loss in batches, update model
    """
    
    def __init__(self, config: STS2GRPOConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.bf16 else torch.float16
        
        print(f"Loading tokenizer: {config.tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        self.lora_adapter_path = str(self.output_dir / "current_lora") if self.config.use_lora else None

        self._setup_environment()
        
        self.global_step = 0
        self.best_reward = float('-inf')
        
        self.generation_log_file = self.output_dir / "generation_log.txt"
        with open(self.generation_log_file, "w", encoding="utf-8") as f:
            f.write(f"STS2 GRPO Generation Log - {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
        
        if self.config.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                config={
                    "model_id": self.config.model_id,
                    "dataset_size": self.config.dataset_size,
                    "num_generations": self.config.num_generations,
                    "max_steps": self.config.max_steps,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.epochs,
                    "beta": self.config.beta,
                    "gamma": self.config.gamma,
                    "omega": self.config.omega,
                    "use_lora": self.config.use_lora,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "use_future_kl": self.config.use_future_kl,
                    "future_kl_tau": self.config.future_kl_tau,
                    "clip_ratio_low": self.config.clip_ratio_low,
                    "clip_ratio_high": self.config.clip_ratio_high,
                    "warmup_ratio": self.config.warmup_ratio,
                },
                tags=["GRPO", "STS2", "RL"]
            )
            print(f"W&B initialized: {wandb.run.name}")
    
    def _setup_model(self, load_adapter_path: str = None):
        """Setup model for training.
        
        Args:
            load_adapter_path: Path to load existing LoRA adapter (optional)
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading model: {self.config.model_id}")
        
        try:
            if self.config.use_lora:
                from peft import LoraConfig, get_peft_model, TaskType, PeftModel
                
                print("Loading reference model...")

                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype,
                    device_map="auto",
                )
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad = False
                
                if load_adapter_path and Path(load_adapter_path).exists():
                    print(f"Loading training model with LoRA adapter from: {load_adapter_path}")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        torch_dtype=self.dtype,
                        device_map="auto",
                    )
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        load_adapter_path,
                        is_trainable=True
                    )
                else:
                    print("Loading training model with new LoRA config...")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        torch_dtype=self.dtype,
                        device_map="auto",
                    )
                    
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.config.lora_r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        target_modules=self.config.lora_target_modules,
                        bias="none",
                    )
                    
                    self.model = get_peft_model(base_model, lora_config)
                
                if self.config.use_gradient_checkpointing:
                    self.model.gradient_checkpointing_enable()
                
                self.model.print_trainable_parameters()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype,
                    device_map="auto",
                )
                
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype,
                    device_map="auto",
                )
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad = False
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
                
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"  Model ID: {self.config.model_id}")
            print(f"  GPU available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
                print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            raise
    
    def _setup_environment(self):
        """Setup STS2 combat environment."""
        from sts2_env.gym_env.combat_env import STS2CombatEnv
        
        self.env = STS2CombatEnv(
            player_hp=self.config.player_hp,
            player_max_hp=self.config.player_max_hp,
            max_turns=self.config.max_steps,
            use_dict_obs=True,
        )
        print(f"Environment: STS2CombatEnv, Player HP: {self.config.player_hp}")
        
        self.rollout_collector = RolloutCollector(
            env=self.env,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
            tokenizer=self.tokenizer,
            lora_adapter_path=self.lora_adapter_path,
        )
    
    def save_transitions(self, transitions: list[Transition], filename: str = "transitions.json"):
        """Save transitions to a local file."""
        transitions_data = []
        for t in transitions:
            transitions_data.append({
                "seed_idx": t.seed_idx,
                "trajectory_id": t.trajectory_id,
                "state_text": t.state_text,
                "action": t.action,
                "step_reward": t.step_reward,
                "final_reward": t.final_reward,
                "format_reward": t.format_reward,
                "total_reward": t.total_reward,
                "trajectory_relative_advantage": t.trajectory_relative_advantage,
                "step_relative_advantage": t.step_relative_advantage,
                "discounted_return": t.discounted_return,
            })
        
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transitions_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(transitions)} transitions to {output_path}")
    
    def save_lora_adapter(self, adapter_dir: str):
        """Save LoRA adapter to a directory for vLLM to load.
        
        Args:
            adapter_dir: Directory path to save the adapter
        """
        if not self.config.use_lora:
            print("Warning: save_lora_adapter called but use_lora is False")
            return
        
        import os
        os.makedirs(adapter_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("Model not loaded. Call _setup_model() first.")
        
        self.model.save_pretrained(adapter_dir)
        print(f"LoRA adapter saved to: {adapter_dir}")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.use_lora:
            self.model.save_pretrained(checkpoint_dir)
            print(f"LoRA adapter saved to: {checkpoint_dir}")
        else:
            torch.save({
                "model": self.model.state_dict(),
            }, checkpoint_dir / "checkpoint.pt")
        
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_reward": self.best_reward,
        }, checkpoint_dir / "trainer_state.pt")
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def create_dataloader(self, transitions: list, advantages: list, batch_size: int, shuffle: bool = True):
        """Create a simple dataloader from transitions and advantages."""
        indices = list(range(len(transitions)))
        
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(transitions), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_transitions = [transitions[j] for j in batch_indices]
            batch_advantages = [advantages[j] for j in batch_indices]
            yield batch_transitions, batch_advantages
    
    def _unload_model(self):
        """Unload model to free memory for vLLM rollout.
        
        Note: optimizer and scheduler are kept to preserve training state.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.ref_model is not None:
            del self.ref_model
            self.ref_model = None
        
        # 清理缓存和强制同步
        gc.collect()
        # torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Model unloaded to free memory (optimizer and scheduler preserved)")
    
    def train(self):
        """Main training loop.
        
        Correct flow (train/inference separation):
        1. Setup model, save LoRA adapter, unload model
        2. Rollout ALL trajectories using vLLM (no model loaded)
        3. Compute advantages for ALL transitions
        4. Setup model, load LoRA adapter, train with batches
        5. Unload model after training
        """
        seed_prompts = ["Win the combat. Defeat all enemies while preserving your HP."] * self.config.dataset_size
        total_rollouts = len(seed_prompts) * self.config.num_generations
        
        estimated_transitions_per_epoch = total_rollouts * (self.config.max_steps // 2)
        total_batches_per_epoch = max(1, estimated_transitions_per_epoch // self.config.batch_size)
        total_optimizer_steps = self.config.epochs * total_batches_per_epoch
        
        warmup_steps = int(total_optimizer_steps * self.config.warmup_ratio)
        
        def cosine_with_warmup(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
            return self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        print("\n[Phase 0] Setup model and save LoRA adapter...")
        self._setup_model()
        self.save_lora_adapter(self.lora_adapter_path)
        self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate
            )
        self.scheduler = LambdaLR(self.optimizer, cosine_with_warmup)
        self._unload_model()
        

        print(f"Warmup steps: {warmup_steps} ({self.config.warmup_ratio*100:.1f}%)")
        
        print("=" * 80)
        print("Starting STS2 GRPO Training")
        print(f"Model: {self.config.model_id}")
        print(f"Dataset size: {self.config.dataset_size}")
        print(f"Num generations per seed: {self.config.num_generations}")
        print(f"Total rollouts per epoch: {total_rollouts}")
        print(f"Max steps per trajectory: {self.config.max_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Warmup steps: {warmup_steps} ({self.config.warmup_ratio*100:.1f}%)")
        print("=" * 80)
                
        for epoch in range(self.config.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print("=" * 80)
            
            seeds = [epoch * len(seed_prompts) + i for i in range(len(seed_prompts))]
            
            print("\n[Phase 1] Collecting rollouts with vLLM...")
            self.rollout_collector.load_vllm()
            all_transitions = self.rollout_collector.collect_rollouts(seed_prompts, seeds=seeds)
            self.rollout_collector.unload_vllm()
            print(f"Collected {len(all_transitions)} transitions")

            
            avg_reward = sum(t.total_reward for t in all_transitions) / max(len(all_transitions), 1)
            print(f"Avg reward: {avg_reward:.4f}")
            
            if self.config.use_wandb:
                import wandb
                import numpy as np
                rewards = [t.total_reward for t in all_transitions]
                step_rewards = [t.step_reward for t in all_transitions]
                final_rewards = [t.final_reward for t in all_transitions]
                format_rewards = [t.format_reward for t in all_transitions]
                
                wandb.log({
                    "rollout/avg_reward": avg_reward,
                    "rollout/num_transitions": len(all_transitions),
                    "rollout/reward_mean": np.mean(rewards),
                    "rollout/reward_std": np.std(rewards),
                    "rollout/reward_min": np.min(rewards),
                    "rollout/reward_max": np.max(rewards),
                    "rollout/step_reward_mean": np.mean(step_rewards),
                    "rollout/final_reward_mean": np.mean(final_rewards),
                    "rollout/format_reward_mean": np.mean(format_rewards),
                    "rollout/epoch": epoch + 1,
                })
            
            print("\n[Phase 2] Computing advantages...")
            results = compute_relative_advantage(
                all_transitions,
                gamma=self.config.gamma,
                omega=self.config.omega,
                use_std_norm=self.config.use_std_norm,
            )
            transitions_list = [r[0] for r in results]
            advantages = [r[1] for r in results]
            print(f"Computed advantages for {len(transitions_list)} transitions")
            
            if self.config.use_wandb:
                import wandb
                import numpy as np
                wandb.log({
                    "advantage/mean": np.mean(advantages),
                    "advantage/std": np.std(advantages),
                    "advantage/min": np.min(advantages),
                    "advantage/max": np.max(advantages),
                    "advantage/epoch": epoch + 1,
                })
            
            print("\n[Phase 3] Setup model for training...")
            
            if self.model is None and self.ref_model is None:
                self._setup_model(load_adapter_path=self.lora_adapter_path)
            
            print("\n[Phase 4] Training with batches...")
            self.model.train()
            num_batches = (len(transitions_list) + self.config.batch_size - 1) // self.config.batch_size
            
            epoch_loss = 0.0
            for batch_idx, (batch_transitions, batch_advantages) in enumerate(
                self.create_dataloader(transitions_list, advantages, self.config.batch_size)
            ):
                loss = compute_loss_batch(
                    model=self.model,
                    ref_model=self.ref_model,
                    batch_transitions=batch_transitions,
                    batch_advantages=batch_advantages,
                    device=self.device,
                    beta=self.config.beta,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_future_kl=self.config.use_future_kl,
                    future_kl_tau=self.config.future_kl_tau,
                    future_kl_clip_low=self.config.future_kl_clip_low,
                    future_kl_clip_high=self.config.future_kl_clip_high,
                    dual_clip_threshold=self.config.dual_clip_threshold,
                    future_kl_clip_high_only=self.config.future_kl_clip_high_only,
                    safety_thresh=self.config.safety_thresh,
                    clip_ratio_low=self.config.clip_ratio_low,
                    clip_ratio_high=self.config.clip_ratio_high,
                )
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                epoch_loss += loss.item()
                
                print(f"  Batch {batch_idx + 1}/{num_batches}: Loss={loss.item():.4f}, "
                      f"LR={self.scheduler.get_last_lr()[0]:.2e}")
                
                if self.config.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                        "train/batch": batch_idx + 1,
                        "train/global_step": self.global_step,
                    })
                
                del batch_transitions, batch_advantages, loss
                gc.collect()
                torch.cuda.empty_cache()
            
            self.model.eval()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch + 1} completed. Avg loss: {avg_loss:.4f}, Avg reward: {avg_reward:.4f}")
            
            if self.config.use_wandb:
                import wandb
                wandb.log({
                    "epoch/avg_loss": avg_loss,
                    "epoch/avg_reward": avg_reward,
                    "epoch/num": epoch + 1,
                    "epoch/best_reward": self.best_reward,
                })
            
            self.save_transitions(all_transitions, f"transitions_epoch_{epoch + 1}.json")
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save_checkpoint("best")
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")
            
            print("\n[Phase 5] Unloading model after training...")
            self.save_lora_adapter(self.lora_adapter_path)
            self._unload_model()
            
            del all_transitions, results, transitions_list, advantages
            gc.collect()
            torch.cuda.empty_cache()
        
        print("\n[Final] Setup model for final checkpoint...")
        if self.model is None:
            self._setup_model(load_adapter_path=self.lora_adapter_path)
        
        self.save_checkpoint("final")
        print("\nTraining completed!")
        
        if self.config.use_wandb:
            import wandb
            wandb.finish()
