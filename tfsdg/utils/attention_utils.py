from dataclasses import dataclass

import torch


@dataclass
class KeyValueTensors(object):
    k: torch.Tensor
    v: torch.Tensor
