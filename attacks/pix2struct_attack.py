"""Custom attack for Pix2Struct."""

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

from attacks.constraints import QuantizationConstraintWithMask

from models.processing.pix2struct_processor import Pix2StructImageProcessor
from models.pix2struct import Pix2StructModel, Pix2StructModelProcessor
from transformers.image_utils import to_numpy_array
from typing import Union, List
from .masks import mask_include_all

# custom modular evasion attack definition
from .modular_attack_grad_accumulation import ModularEvasionAttackFixedEps

class Pix2StructAttack(ModularEvasionAttackFixedEps):
    """Attack implementing a custom forward for Pix2Stuct."""
    def __init__(self, 
                 processor:Pix2StructImageProcessor, 
                 questions: Union[str, List[str]], 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.questions = questions
    
    def forward_loss(self, model:Pix2StructModel, x, target):
        x.requires_grad_(True)
        targets = model.model_processor.reconstruct_targets(target)
        
        total_loss = 0
        for question, answer in zip(self.questions, targets):
            x_header, _, _ = self.processor.my_preprocess_image(x, question)
            patches = self.processor.extract_flattened_patches(x_header)
            _, loss = model.loss_fn(patches.unsqueeze(0), answer.unsqueeze(0))
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

def attack_pix2struct(
        model: Pix2StructModel, 
        processor: Pix2StructImageProcessor,
        image, 
        questions, 
        targets, 
        eps=8.0,
        steps=20,
        step_size=2.0,
        is_targeted=True,
        mask_function=mask_include_all
        ):
    """Compute adversarial perturbation for Pix2Struct model."""
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
    
    attack = Pix2StructAttack(
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
        questions=questions
    )
    test_loader = DataLoader(TensorDataset(input_tensor.unsqueeze(0), labels))

    native_adv_ds = attack(model, test_loader)
    advx, _ = next(iter(native_adv_ds))

    advx = advx.squeeze(0)
    np_img = advx.clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    
    return Image.fromarray(np_img)
