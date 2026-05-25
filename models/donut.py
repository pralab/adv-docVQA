"""Wrapper for the Donut model."""

import torch
from PIL import Image
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from .processing.base_processor import BaseDocVQAProcessor

import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
logger = logging.getLogger(__name__)


class DonutModelProcessor(BaseDocVQAProcessor):
    """Data processing utility for Donut model."""

    def __init__(self) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        self.target_suffix = "</s_answer>"
        self.add_special_tokens = False

    def _process(self, x: Image, q: str) -> torch.Tensor:
        return self.processor(images=x, text=q, return_tensors="pt")

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented."""

    def decode(self, x):
        """Decode model outputs."""
        return self.processor.decode(x, skip_special_tokens=True)

class DonutModel(BaseModel):
    """Wrapper for the DocVQA PyTorch model."""
    MAX_OUTPUT_TOKENS = 50

    def __init__(self, device="cpu") -> None:
        model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
        self._model: torch.nn.Module = (
            AutoModelForVision2Seq.from_pretrained(
                model_name,
            )
        ).to(device)

        self.task_prompt = "<s_docvqa><s_question>{question}</s_question><s_answer>"

        self.model_processor = DonutModelProcessor()
        self._model.config.decoder_start_token_id = self.model_processor.processor.tokenizer.cls_token_id
        self._model.config.pad_token_id = self.model_processor.processor.tokenizer.pad_token_id
        super().__init__()

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return prediction from the Pix2Struct model."""
        return self.decision_function(x)[0]

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self._get_device())
        return self._model.generate(flattened_patches=x)

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


    def loss_fn(self, x: torch.Tensor, target_answer_ids: torch.Tensor, question: str):
        device = self._get_device()
        processor = self.model_processor.processor

        prompt = self.task_prompt.format(question=question)
        prompt_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

        prompt_ids = prompt_ids.to(device)
        target_answer_ids = target_answer_ids.to(device)
        x = x.to(device)
        
        full_decoder_input_ids = torch.cat([
            prompt_ids, 
            target_answer_ids
        ], dim=1)
        
        outputs = self._model(
            pixel_values=x,
            decoder_input_ids=full_decoder_input_ids
        )

        total_loss = outputs.logits.sum() * 0.0
        target_answer_ids = target_answer_ids.squeeze(0)

        for i in range(len(target_answer_ids)):
            logit_position = prompt_ids.shape[1] + -1 + i
            current_logits = outputs.logits[:, logit_position, :]
            
            top1_logit, top1_idx = torch.max(current_logits, dim=-1)
            
            target_id = target_answer_ids[i]
            target_logit = current_logits[0, target_id]
            
            if top1_idx.item() != target_id:
                token_loss = top1_logit.squeeze() - target_logit
                total_loss += token_loss
            else:
                total_loss += 0.0        

        if len(target_answer_ids) > 0:
            total_loss = total_loss / len(target_answer_ids)

        # self._debug_token_optimization(x=x,
        #                                total_loss=total_loss,
        #                                target_answer_ids=target_answer_ids,
        #                                prompt_ids=prompt_ids,
        #                                outputs=outputs)

        return outputs, total_loss

    def _debug_token_optimization(self, x, total_loss, target_answer_ids, prompt_ids, outputs):
        processor = self.model_processor.processor
        device = self._get_device()
        
        self._model.eval()
        with torch.no_grad():
            logger.info(f"###### Debug #####")
            logger.info(f"Loss: {total_loss.item():.4f}")
            
            # check each token in the target sequence, print the top k=5
            k = 5
            logger.info(f'\n top {k} TOKENS')
            for i in range(len(target_answer_ids)):
                logger.info(f"\n Token target n.{i} - {processor.decode(target_answer_ids[i])}")
                logit_position = prompt_ids.shape[1] - 1 + i
                step_logits = outputs.logits[:, logit_position, :]

                probs = torch.nn.functional.softmax(step_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, k=k)

                for j in range(k):
                    token_id = top_k_indices[0, j].item()
                    token_prob = top_k_probs[0, j].item()
                    decoded_token = processor.decode([token_id])
                    logger.info(f"\t\t{j+1}°: token='{decoded_token}', prob={token_prob:.4f}")

            generated_outputs = self._model.generate(
                pixel_values=x.to(device),
                decoder_input_ids=prompt_ids,
                max_length=self.MAX_OUTPUT_TOKENS,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_outputs, skip_special_tokens=False)[0]
            logger.info(f"GENERATED = {generated_text}")
            logger.info("#" * 40)

    def train(self, dataloader: DataLoader) -> BaseModel:
        """Not implemented."""

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input patches.

        Returns
        -------
        torch.Tensor
            Output from the model.
        """
        x = x.to(device=self._get_device())
        return self._decision_function(x)

    def torch_predict(self, image, questions):
        device = self._get_device()
        processor = self.model_processor.processor
        
        prompts = [self.task_prompt.format(question=q) for q in questions]
        inputs = processor(images=[image]*len(questions),
                                text=prompts, 
                                return_tensors="pt",
                                padding=True).to(device)

        self._model.eval()
        outputs = self._model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            max_length=self.MAX_OUTPUT_TOKENS,
            do_sample=False
        )

        # generate the answer and filter out special tokens
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=False)
        answers = []
        for text in generated_texts:
            clean_text = text.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "")

            if "<s_answer>" in clean_text:
                answer_part = clean_text.split("<s_answer>", 1)[1]
                answer = answer_part.split("</s_answer>", 1)[0]
                answers.append(answer.strip())
            else:
                print("*********************** DONUT ERROR")
                print("Answer without <s_answer>")
                print(f"Clean text = {clean_text}")
                print("*********************** DONUT ERROR")
                print("",flush=True)

        return answers
