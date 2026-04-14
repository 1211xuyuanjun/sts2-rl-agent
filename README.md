# STS2 GRPO Trainer

基于 GiGPO (Grandmaster-Informed GRPO) 算法的 STS2 强化学习训练框架。

## 项目简介

本项目实现了一个基于 GRPO (Group Relative Policy Optimization) 的强化学习训练系统，用于训练 STS2 (Slay the Spire 2) 游戏代理。使用了 LoRA 高效微调技术和 vLLM 加速推理。

## 功能特性

- **GiGPO 算法**：结合了 Grandmaster-informed 思想的 GRPO 变体
- **LoRA 高效微调**：使用低秩适配器进行参数高效微调
- **vLLM 加速推理**：使用 vLLM 进行快速批量推理
- **Future-KL (FIPO)**：可选的未来 KL 散度优化
- **WandB 集成**：训练指标实时可视化
- **灵活配置**：支持自定义超参数和训练策略

## 项目结构

```
train/gigpo/
├── main.py           # 训练入口
├── trainer.py        # 训练器核心逻辑
├── config.py         # 配置类
├── rollout.py        # Rollout 收集器
├── loss.py          # 损失函数计算
├── advantage.py      # Advantage 计算
├── inference.py      # 推理脚本
├── prompts.py       # 提示词模板
├── transition.py    # Transition 数据结构
├── fipo.py          # Future-KL 实现
└── __init__.py      # 包初始化
```

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install torch transformers peft vllm tqdm wandb

# 确保安装了 sts2_env 环境
pip install -e sts2_env
```

### 训练

```bash
# 基本训练
python -m train.gigpo.main \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir outputs/sts2_grpo \
    --dataset-size 50 \
    --num-generations 4 \
    --epochs 3
```

### 推理评估

```bash
# 使用训练好的 LoRA adapter 进行推理
python -m train.gigpo.inference \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-dir outputs/sts2_grpo/current_lora \
    --num-episodes 10
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-id` | Qwen/Qwen2.5-1.5B-Instruct | 模型 ID |
| `--dataset-size` | 50 | 数据集大小 |
| `--num-generations` | 4 | 每个 seed 的生成数 |
| `--batch-size` | 2 | 批次大小 |
| `--learning-rate` | 1e-5 | 学习率 |
| `--epochs` | 3 | 训练轮数 |
| `--use-lora` | True | 是否使用 LoRA |
| `--lora-r` | 16 | LoRA 秩 |
| `--use-vllm` | False | 是否使用 vLLM 推理 |
| `--use-future-kl` | False | 是否使用 Future-KL |
| `--use-wandb` | True | 是否使用 WandB |

## 算法说明

### GRPO

GRPO (Group Relative Policy Optimization) 是一种策略梯度优化方法，对每个状态采样多个动作，计算动作的相对优势。

### GiGPO

GiGPO 在 GRPO 的基础上增加了：
- **Trajectory-level Advantage**：轨迹级别的优势估计
- **Step-level Advantage**：步骤级别的优势估计（通过 omega 参数平衡）
- **Std Normalization**：对优势进行标准化处理

### Future-KL (FIPO)

Future-KL 是一种基于未来信息的 KL 散度优化方法，可以更稳定地控制策略更新。

## 奖励设计

- **Step Reward**：每步获得的即时奖励
- **Final Reward**：基于最终 HP 的奖励（HP比例 或 -1 如果死亡）
- **Format Reward**：动作格式正确性的奖励

## 输出目录

训练过程中会生成以下文件：

```
outputs/sts2_grpo/
├── current_lora/           # 当前 LoRA adapter
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── best/                   # 最佳模型检查点
├── epoch_*/                # 每 N 轮的检查点
├── transitions_epoch_*.json # 每轮的 transition 数据
├── generation_log.txt      # 生成日志
└── trainer_state.pt        # 训练器状态
```

## WandB 监控

启用 WandB 后，可以实时监控以下指标：

- `train/loss`：训练损失
- `train/learning_rate`：学习率
- `epoch/avg_loss`：平均损失
- `epoch/avg_reward`：平均奖励
- `rollout/reward_*`：奖励分布统计
- `advantage/*`：Advantage 统计

## 示例命令

### 使用 LoRA 训练

```bash
python -m train.gigpo.main \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --dataset-size 100 \
    --num-generations 8 \
    --epochs 5
```

### 使用 Future-KL 训练

```bash
python -m train.gigpo.main \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --use-future-kl \
    --future-kl-tau 32.0 \
    --beta 0.02
```

### 评估多个 episodes

```bash
python -m train.gigpo.inference \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-dir outputs/sts2_grpo/best \
    --num-episodes 20 \
    --seed 0
```

## 注意事项

1. **GPU 内存**：训练需要足够的 GPU 内存，建议使用 24GB+ 的 GPU
2. **vLLM 首次运行**：vLLM 首次加载会编译 CUDA kernels，可能较慢
3. **LoRA adapter 路径**：确保在训练前保存了 LoRA adapter

## 许可证

MIT License
