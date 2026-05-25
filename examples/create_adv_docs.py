import os
import sys
import argparse
import logging
import torch
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.model_registry import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttackScenarioRunner:
    def __init__(self, model, processor, attack_fn, mask_fn, doc_path, model_name, attack_kwargs):
        self.model = model
        self.processor = processor
        self.attack_fn = attack_fn
        self.mask_fn = mask_fn
        self.doc_path = doc_path
        self.model_name = model_name
        self.attack_kwargs = attack_kwargs
        
        self.image = Image.open(self.doc_path).convert("RGB")

    def run_scenario(self, questions: list | str, targets: list | str, is_targeted: bool):        
        adv_example = self.attack_fn(
            model=self.model,
            processor=self.processor,
            image=self.image,
            questions=questions,
            targets=targets,
            is_targeted=is_targeted,
            mask_function=self.mask_fn,
            **self.attack_kwargs
        )
        
        out_path = os.path.join(os.path.dirname(__file__), f"{self.model_name}_optimized.jpg")
        adv_example.save(out_path)
        logger.info(f"Saved to: {out_path}")

        logger.info("Running inference...")
        q_list = [questions] if isinstance(questions, str) else questions 
        
        y_pred = self.model.torch_predict(self.image, q_list)
        y_adv = self.model.torch_predict(adv_example, q_list)
        
        target_label = "[target]" if is_targeted else "[GT]"
        logger.info(f"{target_label} : {targets}")
        logger.info(f"[y_pred]   : {y_pred}")
        logger.info(f"[y_adv]: {y_adv}\n")

def get_args():
    parser = argparse.ArgumentParser(description="advDocVQA")
    parser.add_argument("--model", type=str, choices=["donut", "pix2struct"], required=True, help="Target model to attack.")
    parser.add_argument("--scenario", type=str, choices=["single", "multiple", "doa", "all"], default="single",help="Attack scenario to execute.")
    parser.add_argument("--doc_name", type=str, default="0a0a0792728288619a600f55_0.jpg",help="Filename of the document.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (e.g., cuda, cpu).")
    parser.add_argument("--mask", type=str, default="include_all", help="Mask type to use. Must match a key in config.AVAILABLE_MASKS, e.g., 'include_all' or 'bottom_right_corner'.")
    parser.add_argument("--eps", type=float, default=None, help="Perturbation magnitude.")
    parser.add_argument("--steps", type=int, default=None, help="Number of PGD steps.")
    parser.add_argument("--step_size", type=float, default=None, help="Step size.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    doc_path = os.path.join(os.path.dirname(__file__), "documents", args.doc_name)
    if not os.path.exists(doc_path):
        logger.error(f"Doc not found at: {doc_path}")
        sys.exit(1)

    logger.info(f"Init {args.model} on {args.device}...")
    processor, model, attack_fn, mask_fn = get_model(args.model, args)

    attack_kwargs = {}
    if args.eps is not None: attack_kwargs['eps'] = args.eps
    if args.steps is not None: attack_kwargs['steps'] = args.steps
    if args.step_size is not None: attack_kwargs['step_size'] = args.step_size

    runner = AttackScenarioRunner(model, processor, attack_fn, mask_fn, doc_path, args.model, attack_kwargs)

    if args.scenario in ["single", "all"]:
        runner.run_scenario(
            questions="How much is the total?",
            targets="$0.00",
            is_targeted=True
        )
    
    if args.scenario in ["multiple", "all"]:
        runner.run_scenario(
            questions=["How much is the total?", "What is the invoice number?"],
            targets=["$0.00", "0"],
            is_targeted=True
        )
    
    if args.scenario in ["doa", "all"]:
        runner.run_scenario(
            questions=["How much is the total?", "What is the invoice number?", "Are there comments?"],
            targets=["$1,827.50", "8257383", "brad philips"],
            is_targeted=False
        )