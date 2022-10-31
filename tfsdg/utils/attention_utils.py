from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class KeyValueTensors(object):
    k: Union[torch.Tensor, List[torch.Tensor]]
    v: Union[torch.Tensor, List[torch.Tensor]]
