"""
Exponential Moving Average (EMA) for model weights.
Used by effb3 branch.
"""

import torch
from torch import nn


class ModelEMA:
    """Exponential Moving Average of model weights.

    Args:
        model: nn.Module
        decay: EMA decay rate (default 0.999)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].lerp_(v, 1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}
