"""Constraints for implementing the attack."""

import torch
from typing import Union
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.optimization.constraints import QuantizationConstraint
from secmlt.optimization.constraints import InputSpaceConstraint

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

class QuantizationConstraint(InputSpaceConstraint):
    """Constraint for ensuring quantized outputs into specified levels."""

    def __init__(
        self,
        preprocessing: DataProcessing = None,
        levels: Union[list[float], torch.Tensor, int] = 255,
    ) -> None:
        """
        Create the QuantizationConstraint.

        Parameters
        ----------
        preprocessing: DataProcessing
            Preprocessing to apply the constraint in the input space.
        levels : int, list[float] | torch.Tensor
            Number of levels or specified levels.
        """
        if isinstance(levels, int | float):
            if levels < 2:  # noqa: PLR2004
                msg = "Number of levels must be at least 2."
                raise ValueError(msg)
            if int(levels) != levels:
                msg = "Pass an integer number of levels."
                raise ValueError(msg)
            # create uniform levels if an integer is provided
            self.levels = torch.linspace(0, 1, int(levels))
        elif isinstance(levels, list):
            self.levels = torch.tensor(levels, dtype=torch.float32)
        elif isinstance(levels, torch.Tensor):
            self.levels = levels.type(torch.float32)
            if len(self.levels) < 2:  # noqa: PLR2004
                msg = "Number of custom levels must be at least 2."
                raise ValueError(msg)
        else:
            msg = "Levels must be an integer, list, or torch.Tensor."
            raise TypeError(msg)
        # sort levels to ensure they are in ascending order
        self.levels = self.levels.sort().values  # noqa: PD011
        super().__init__(preprocessing)

    def old_apply_constraint(self, x: torch.Tensor) -> torch.Tensor:
        # reshape x to facilitate broadcasting with custom levels
        x_expanded = x.unsqueeze(-1)
        # calculate the absolute difference between x and each custom level
        distances = torch.abs(x_expanded - self.levels)
        # find the index of the closest custom level
        nearest_indices = torch.argmin(distances, dim=-1)
        # quantize x to the nearest custom level
        return self.levels[nearest_indices]
    
    def _apply_constraint(self, x, *args, **kwargs):
        min_val, max_val = self.levels[0].item(), self.levels[-1].item()
        x_clamped = x.clamp(min_val, max_val)
        idx = torch.round(x_clamped).long()
        return self.levels[idx]

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