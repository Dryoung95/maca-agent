import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, a, b):
        # Single fused kernel: out = alpha * a + b
        # Replaces two full-tensor memory passes (scale, then add) with one.
        return cuda_extension.axpby_forward(a, b, self.alpha, config=0)
