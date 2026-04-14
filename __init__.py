"""
GiGPO (Group-in-Group Policy Optimization) for Slay the Spire 2.

Modules:
- config: Configuration class
- prompts: Prompt templates and functions
- transition: Transition data class
- rollout: Rollout collection methods
- advantage: Advantage computation methods
- loss: Loss computation methods
- trainer: Main trainer class
"""

from .config import STS2GRPOConfig
from .transition import Transition
from .trainer import STS2GRPOTrainer

__all__ = [
    "STS2GRPOConfig",
    "Transition",
    "STS2GRPOTrainer",
]
