"""定义模型。"""
import torch
from torch import nn

import config


def build_model():
    torch.manual_seed(config.SEED)

    model = nn.Sequential(
        nn.Linear(len(config.FEATURE_COLUMNS), config.HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(config.HIDDEN_SIZE, 1),
    )
    model = model.to(config.DEVICE)
    return model
