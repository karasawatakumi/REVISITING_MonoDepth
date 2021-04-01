import argparse
from typing import Optional

import torch
from omegaconf import DictConfig

from src.data import get_test_loader
from src.models import build_model
from src.engine import test
from src.utils import load_config, make_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test a predictor')
    parser.add_argument('ckpt', type=str, help='the checkpoint file')
    parser.add_argument('--config', type=str, default=None, help='train config file path')
    parser.add_argument('--show-dir', type=str, default=None,
                        help='Please specify the directory if you want to save predicted figures.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Overwrite configs. (ex. SOLVER.NUM_WORKERS=8)')
    return parser.parse_args()


def main(config: DictConfig, ckpt: dict, show_dir: Optional[str] = None):

    # seed
    if config.SEED is not None:
        make_deterministic(seed=config.SEED)

    # data
    test_loader = get_test_loader(config)

    # model
    model = build_model(config, model_state_dict=ckpt['model_state_dict'])

    # test
    test(model=model,
         data_loader=test_loader,
         device=config.DEVICE,
         threshold_edge=config.TEST.THRESHOLD_EDGE,
         show_dir=show_dir)


if __name__ == "__main__":
    args = parse_args()

    # load config, ckpt
    config = load_config(args.config, update_dotlist=args.opts)
    ckpt: dict = torch.load(args.ckpt)

    main(config, ckpt, args.show_dir)
