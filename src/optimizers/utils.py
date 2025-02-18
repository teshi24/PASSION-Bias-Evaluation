from typing import Callable

import numpy as np
import torch


def get_optimizer_type(optimizer_name: str) -> Callable:
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    optimizer_cls = optimizer_dict.get(optimizer_name, np.nan)
    if optimizer_cls is np.nan:
        raise ValueError("Invalid optimizer name.")
    return optimizer_cls
