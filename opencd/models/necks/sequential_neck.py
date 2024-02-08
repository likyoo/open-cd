# Copyright (c) Open-CD. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule

from opencd.registry import MODELS


@MODELS.register_module()
class SequentialNeck(BaseModule):
    def __init__(self, necks):
        super().__init__()
        self.necks = nn.ModuleList()
        for neck in necks:
            self.necks.append(MODELS.build(neck))

    def forward(self, *args, **kwargs):
        for neck in self.necks:
            args = neck(*args, **kwargs)
        return args