from .pix2struct import Pix2StructModel
from .processing.pix2struct_processor import Pix2StructImageProcessor
from attacks.pix2struct_attack import attack_pix2struct

from .donut import DonutModel
from .processing.donut_processor import DonutImageProcessor
from attacks.donut_attack import attack_donut
from attacks.masks import mask_include_all, mask_bottom_right_corner

MODEL_NAMES = ['pix2struct','donut']

AVAILABLE_MASKS = {
    "include_all": mask_include_all,
    "bottom_right_corner": mask_bottom_right_corner
}

def get_model(name:str, args):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}. Available models: {MODEL_NAMES}")

    # Initialize the model, processor, and attack based on the model name
    if name == 'pix2struct':
        processor = Pix2StructImageProcessor()
        model = Pix2StructModel(device=args.device)
        attack = attack_pix2struct

    elif name == 'donut':
        processor = DonutImageProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", use_fast=True)
        model = DonutModel(device=args.device)
        attack = attack_donut
    
    # Retrive the mask function ptr based on available masks
    if args.mask not in AVAILABLE_MASKS:
        raise ValueError(f"Unknown mask: {args.mask}. Available masks: {list(AVAILABLE_MASKS.keys())}")
    mask_function = AVAILABLE_MASKS[args.mask]

    return processor, model, attack, mask_function