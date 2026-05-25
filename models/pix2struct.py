"""Wrapper for the Pix2Struct model."""

import torch
from PIL import Image
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from typing import Union, List
from .processing.base_processor import BaseDocVQAProcessor

class Pix2StructModelProcessor(BaseDocVQAProcessor):
    """Data processing utility for Pix2Struct model."""

    def __init__(self) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")
        self.target_suffix = ""
        self.add_special_tokens = True

    def _process(self, x: Image, q: str) -> torch.Tensor:
        return self.processor(images=x, text=q, return_tensors="pt")

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented."""

    def decode(self, x):
        """Decode model outputs."""
        return self.processor.decode(x, skip_special_tokens=True)

class Pix2StructModel(BaseModel):
    """Wrapper for the DocVQA PyTorch model."""
    MAX_OUTPUT_TOKENS = 50

    def __init__(self, device="cpu") -> None:
        self._model: torch.nn.Module = (
            Pix2StructForConditionalGeneration.from_pretrained(
                "google/pix2struct-docvqa-base",
            )
        ).to(device)
        self.model_processor = Pix2StructModelProcessor()
        super().__init__()

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return prediction from the Pix2Struct model."""
        return self.decision_function(x)[0]

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented"""

    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        Return the gradient of the loss w.r.t. the input.

        Parameters
        ----------
        x : torch.Tensor
            Input patches.
        y : int
            Target answer.

        Returns
        -------
        torch.Tensor
            Gradient of the loss w.r.t. x.
        """
        x = x.to(device=self._get_device())
        x.requires_grad = True
        loss = self.loss_fn(x=x, y=y)
        loss.backward()
        return x.grad

    def loss_fn(self, x: torch.Tensor, y: str):
        """
        Compute loss function.

        Parameters
        ----------
        x : torch.Tensor
            Input patches.
        y : str
            Target answer.

        Returns
        -------
        predictions: torch.Tensor
            The predictions from the model.
        loss: torch.Tensor
            The loss.
        """
        x = x.to(device=self._get_device())
        y = y.to(device=self._get_device())
        predictions = self._model(
            labels=y,
            flattened_patches=x,
        )
        return predictions, predictions.loss

    def train(self, dataloader: DataLoader) -> BaseModel:
        """Not implemented."""
    
    def torch_predict(self, image, questions):
        device = self._get_device()
        processor = self.model_processor.processor
        inputs = processor(images=[image]*len(questions), text=questions, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.MAX_OUTPUT_TOKENS)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_texts