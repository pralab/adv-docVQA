import torch
from typing import Tuple

def mask_include_all(image: torch.Tensor):
    """
    Creates a mask for the image, including all pixels.
    """
    mask = torch.ones_like(image, dtype=torch.bool)
    return mask

def mask_bottom_right_corner(image: torch.Tensor, ratio=0.15) -> torch.Tensor:
    """
    Creates a mask for the image on the bottom right corner.
    """
    mask = torch.zeros_like(image, dtype=torch.bool)
    H, W, _ = image.shape
    size = int(min(W,H)*ratio)
    x_start, y_start = W - size, H - size
    mask[y_start:, x_start:, :] = 1  # Exclude the bottom-right corner
    return mask