# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.utils import resize
from mmseg.models.decode_heads.segformer_head import SegformerHead

from opencd.registry import MODELS

@MODELS.register_module()
class MLPSegHead(SegformerHead):
    def __init__(self, out_size, **kwargs):
        super().__init__(**kwargs)

        self.out_size = out_size
        
    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=self.out_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        return out