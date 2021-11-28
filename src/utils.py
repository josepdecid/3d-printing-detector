import random

import numpy as np
import torch


def set_seeds(value: int) -> None:
    torch.manual_seed(value)
    random.seed(value)
    np.random.seed(value)
