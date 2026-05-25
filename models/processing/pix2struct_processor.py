import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union
import io

from transformers.image_transforms import to_pil_image, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
import textwrap
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

DEFAULT_FONT_PATH = "ybelkada/fonts"

def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

def render_header(
    image: np.ndarray, 
    header: str, 
    input_data_format: Optional[Union[str, ChildProcessError]] = None, 
    **kwargs
):
    """
    Renders the input text as a header on the input image.

    Args:
        image (`np.ndarray`):
            The image to render the header on.
        header (`str`):
            The header text.
        data_format (`Union[ChannelDimension, str]`, *optional*):
            The data format of the image. Can be either "ChannelDimension.channels_first" or
            "ChannelDimension.channels_last".

    Returns:
        `np.ndarray`: The image with the header rendered.
    """

    # Convert to PIL image if necessary
    image = to_pil_image(image, input_data_format=input_data_format)

    header_image = render_text(header, **kwargs)
    new_width = max(header_image.width, image.width)

    new_height = int(image.height * (new_width / image.width))
    new_header_height = int(header_image.height * (new_width / header_image.width))

    new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

    # Convert back to the original framework if necessary
    new_image = to_numpy_array(new_image)

    if infer_channel_dimension_format(new_image) == ChannelDimension.LAST:
        new_image = to_channel_dimension_format(new_image, ChannelDimension.LAST)

    return new_image

def render_text(
    text: str,
    text_size: int = 36,
    text_color: str = "black",
    background_color: str = "white",
    left_padding: int = 5,
    right_padding: int = 5,
    top_padding: int = 5,
    bottom_padding: int = 5,
    font_bytes: Optional[bytes] = None,
    font_path: Optional[str] = None,
) -> Image.Image:
    # Add new lines so that each line is no more than 80 characters.
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = "\n".join(lines)

    if font_bytes is not None and font_path is None:
        font = io.BytesIO(font_bytes)
    elif font_path is not None:
        font = font_path
    else:
        font = hf_hub_download(DEFAULT_FONT_PATH, "Arial.TTF")
    font = ImageFont.truetype(font, encoding="UTF-8", size=text_size)

    # Use a temporary canvas to determine the width and height in pixels when
    # rendering the text.
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    # Create the actual image with a bit of padding around the text.
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)
    return image

class Pix2StructImageProcessor:
    def __init__(
            self, 
            patch_size={"height": 16, "width": 16},
            max_patches=2048,
            is_vqa : bool = True
            ) -> None:
        self.patch_size = patch_size
        self.max_patches = max_patches

    def normalize(
            self, 
            image: np.ndarray
            ) -> np.ndarray:
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))
        return (image - mean) / adjusted_stddev, mean, std

    def my_render_header(
        self,
        image_tensor: torch.Tensor, 
        header: str, 
        input_data_format: Optional[Union[str, ChildProcessError]] = None, 
        **kwargs  
        ):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor.permute(2, 0, 1)
        C, H, W  = image_tensor.shape

        header_image = render_text(header, **kwargs)
        numpy_header = np.array(header_image)
        header_tensor = torch.from_numpy(numpy_header).float().permute(2, 0, 1) 

        if header_tensor.dim() == 4:
            header_tensor = header_tensor.squeeze(0)

        _, header_H, header_W = header_tensor.shape
        new_width = max(W, header_W)
        
        new_height = int(H * (new_width / W))
        new_header_height = int(header_H * (new_width / header_W))

        image_resized = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )

        header_resized = F.interpolate(
            header_tensor.unsqueeze(0),
            size=(new_header_height, new_width),
            mode='bilinear',
            align_corners=False
        )

        image_resized = image_resized.squeeze(0)
        header_resized = header_resized.squeeze(0)
        final_image = torch.cat([header_resized, image_resized], dim=1)

        return final_image

    def extract_flattened_patches(
            self,
            image: torch.Tensor,
            max_patches: int = 2048,
            patch_size: dict = {"height": 16, "width": 16},
            ) -> torch.Tensor:

        patch_height, patch_width  = patch_size["height"] ,patch_size["width"]
        _, height, width = image.shape

        scale = math.sqrt(max_patches * (patch_height / height) * (patch_width / width))
        num_rows = max(min(math.floor(scale * height / patch_height), max_patches), 1)
        num_cols = max(min(math.floor(scale * width / patch_width), max_patches), 1)

        resized_height = max(num_rows * patch_height, 1)
        resized_width = max(num_cols * patch_width, 1)
        
        # resize operation
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(0)
        
        patches = torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # Offset by 1 so the ids do not contain zeros, which represent padding.
        row_ids += 1
        col_ids += 1

        # Prepare additional patch features.
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        return result
    
    def my_normalize(
        self,
        image_tensor: torch.Tensor
        ) -> torch.Tensor:
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32)
        mean = torch.mean(image_tensor)
        std = torch.std(image_tensor)
        adjusted_stddev = max(std.item(), 1.0 / math.sqrt(image_tensor.numel()))
        normalized = (image_tensor - mean) / adjusted_stddev
        return normalized, mean, std
    
    def preprocess(
            self,
            image: Image.Image,
            header_text: Optional[str] = None,
            requires_grad=False
            ) -> torch.Tensor:
        image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        
        # apply pre processing: header rendering, normalization and then extract flattened patches
        image = render_header(image,header_text)
        image = self.normalize(image)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        image.requires_grad = requires_grad
        return image, self.extract_flattened_patches(image)
    
    def preprocess_image(
            self,
            image: Image.Image,
            header_text: Optional[str] = None,
            ) -> torch.Tensor:
        image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        
        # apply pre processing: header rendering, normalization and then extract flattened patches
        image = render_header(image,header_text)
        #image, mean, std = self.normalize(image)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        #return image, mean, std
        return image, 0,0
    
    def my_preprocess_image(
            self,
            image: torch.Tensor,
            header_text: Optional[str] = None,
            ) -> torch.Tensor:
        image = self.my_render_header(image,header_text)
        image, mean, std = self.my_normalize(image)
        return image, mean, std