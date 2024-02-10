# Copyright (c) Open-CD. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS


@MODELS.register_module()
class DS_FPNHead(BaseDecodeHead):
    """LightCDNet: Lightweight Change Detection Network Based 
    on VHR Images.

    This head is the implementation of `LightCDNet
    <https://ieeexplore.ieee.org/document/10214556>`_.

    """

    def __init__(self, **kwargs):
        super(DS_FPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.scale_heads = nn.ModuleList()
        for i in range(len(self.in_channels)):
            scale_head = []
            scale_head.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):
        inputs = inputs[1:]
        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.in_channels)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)

        return output
