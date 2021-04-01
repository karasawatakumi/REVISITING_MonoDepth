from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig

from . import modules, net, resnet, densenet, senet


def build_model(config: DictConfig, model_state_dict: Optional[dict] = None) -> nn.Module:
    model_type = config.MODEL.NAME

    if model_type == 'resnet':
        original_model = resnet.resnet50(pretrained=True)
        encoder = modules.E_resnet(original_model)
        model = net.model(encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    elif model_type == 'densenet':
        original_model = densenet.densenet161(pretrained=True)
        encoder = modules.E_densenet(original_model)
        model = net.model(encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    elif model_type == 'senet':
        original_model = senet.senet154(pretrained='imagenet')
        encoder = modules.E_senet(original_model)
        model = net.model(encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    else:
        raise NotImplementedError

    model.to(config.DEVICE)

    # load
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    return model
