"""Constraints for implementing the attack."""

import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.optimization.constraints import QuantizationConstraint

class MeanStdDataProcessing(DataProcessing):
    """MeanStd data processing."""

    def __init__(self, mean=0.0, std=1.0) -> None:
        self.mean = mean
        self.std = std
        super().__init__()

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x -= self.mean
        x /= self.std
        return x

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        x *= self.std
        x += self.mean
        return x

class QuantizationConstraintWithMask(QuantizationConstraint):
    """
    Quantization constraint with masked components (that will not be quantized).

    Required, e.g., to not quantize the position encoding.
    """

    def __init__(self, preprocessing=None, levels=255, mask=None) -> None:
        self.mask = mask
        super().__init__(preprocessing=preprocessing, levels=levels)

    def __call__(self, x):
        """Apply the quantization."""
        transformed_x = x.detach().clone()
        if self.mask is None:
            self.mask = torch.ones_like(x)
        transformed_x = super().__call__(transformed_x)
        return torch.where(self.mask, transformed_x, x)
