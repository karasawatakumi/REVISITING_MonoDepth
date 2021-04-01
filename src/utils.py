import dataclasses
import random
from typing import Optional, Tuple, List

import torch
import numpy as np
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .models import build_model


def load_config(cfg_path: Optional[str] = None,
                default_cfg_path: str = 'configs/default.yaml',
                update_dotlist: Optional[List[str]] = None) -> DictConfig:

    config = OmegaConf.load(default_cfg_path)
    if cfg_path is not None:
        optional_config = OmegaConf.load(cfg_path)
        config = OmegaConf.merge(config, optional_config)
    if update_dotlist is not None:
        update_config = OmegaConf.from_dotlist(update_dotlist)
        config = OmegaConf.merge(config, update_config)

    OmegaConf.set_readonly(config, True)

    return config


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_training_modules(config: DictConfig, resume_from: Optional[str] = None
                             ) -> Tuple[int, nn.Module, Optimizer, _LRScheduler]:

    model: nn.Module = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.SOLVER.BASE_LR,
                                 weight_decay=config.SOLVER.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.SOLVER.LR_STEP_SIZE,
        config.SOLVER.LR_GAMMA,
    )

    # resume
    start_epoch = 0
    if resume_from is not None:
        ckpt: dict = torch.load(resume_from)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']

    return start_epoch, model, optimizer, scheduler
