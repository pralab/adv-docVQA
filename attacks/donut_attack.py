"""Custom attack for Donut."""

import numpy as np
import torch
from PIL import Image
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    LInfConstraint,
    MaskConstraint,
)
from secmlt.optimization.gradient_processing import (
    LinearProjectionGradientProcessing,
)
from secmlt.optimization.initializer import Initializer
from torch.utils.data import DataLoader, TensorDataset

from attacks.constraints import  QuantizationConstraintWithMask
from models.processing.donut_processor import DonutImageProcessor
from models.donut import DonutModel, DonutModelProcessor
from transformers.image_utils import to_numpy_array
from typing import Union, List
from .masks import mask_include_all

from .modular_attack_grad_accumulation import ModularEvasionAttackFixedEps


class DonutAttack(ModularEvasionAttackFixedEps):
    """Attack implementing a custom forward for Pix2Stuct."""
    def __init__(self,
                 processor: DonutImageProcessor,
                 questions: Union[str, List[str]],
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.questions = questions

    def forward_loss(self, model: DonutModel, x, target):
        """Compute the loss for the given input."""
        x.requires_grad_(True)
        targets = model.model_processor.reconstruct_targets(target)

        total_loss = 0
        for y, question in zip(targets, self.questions):
            x_preproc = self.processor(x.squeeze(0))
            x_input = x_preproc["pixel_values"][0]
            _, loss = model.loss_fn(x_input.unsqueeze(0), y.unsqueeze(0), question)
            loss /= len(self.questions)
            loss.backward()
            total_loss += loss.item()

        return None, total_loss
  
def questions_target_validation(questions, targets):
    assert (isinstance(questions, str) and isinstance(targets, str)) or \
           (isinstance(questions, list) and isinstance(targets, list) and len(questions) == len(targets)) or \
           (isinstance(questions, list) or isinstance(questions, str) and targets is None), \
           "Both questions and targets must either be both strings or both lists of the same length."
    
    if isinstance(questions, str) and isinstance(targets, str):
        questions, targets = [questions], [targets]
    
    return questions, targets

def attack_donut(
        model, 
        processor: DonutImageProcessor,
        image, 
        questions, 
        targets,
        eps=32.0,
        steps=100,
        step_size=2.0,
        is_targeted=True,
        mask_function=mask_include_all
        ):
    """Compute adversarial perturbation for Donut model."""
    questions, targets = questions_target_validation(questions, targets)

    input_tensor = torch.tensor(to_numpy_array(image).astype(np.float32), requires_grad=True)
    labels = model.model_processor.get_input_ids(targets)

    perturbation_mask = mask_function(input_tensor)

    perturbation_constraints = [
        MaskConstraint(mask=perturbation_mask),
        LInfConstraint(radius=float(eps)),
    ]
    domain_constraints = [
        QuantizationConstraintWithMask(
            mask=perturbation_mask.unsqueeze(0),
            levels=torch.arange(0, 256),
        ),
    ]
    gradient_processing = LinearProjectionGradientProcessing(LpPerturbationModels.LINF)
    
    attack = DonutAttack(
        y_target=labels if is_targeted else None,
        num_steps=int(steps),
        step_size=float(step_size),
        loss_function=model.loss_fn,
        optimizer_cls=torch.optim.Adam,
        manipulation_function=AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        ),
        initializer=Initializer(),
        gradient_processing=gradient_processing,
        processor = processor,
        questions = questions
    )
    test_loader = DataLoader(TensorDataset(input_tensor.unsqueeze(0), labels))

    native_adv_ds = attack(model, test_loader)
    advx, _ = next(iter(native_adv_ds))

    advx = advx.squeeze(0)
    np_img = advx.clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    
    return Image.fromarray(np_img)