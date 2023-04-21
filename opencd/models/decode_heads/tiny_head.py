# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS


@MODELS.register_module()
class TinyHead(BaseDecodeHead):
    """
    This head is the implementation of `TinyCDv2
    <https://arxiv.org/abs/>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
        priori_attn (bool): Whether use Priori Guiding Connection.
            Default to False.
    """

    def __init__(self, feature_strides, priori_attn=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        if priori_attn:
            attn_channels = self.in_channels[0]
            self.in_channels = self.in_channels[1:]
            feature_strides = feature_strides[1:]
        self.feature_strides = feature_strides
        self.priori_attn = priori_attn


        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
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

        if self.priori_attn:
            self.gen_diff_attn = ConvModule(
                in_channels=attn_channels // 2,
                out_channels=self.channels,
                kernel_size=1,
                stride=1,
                groups=1,
                norm_cfg=None,
                act_cfg=None
            )

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        if self.priori_attn:
            early_x = x[0]
            x = x[1:]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if self.priori_attn:
            x1_, x2_ = torch.chunk(early_x, 2, dim=1)
            diff_x = torch.abs(x1_ - x2_)
            diff_x = self.gen_diff_attn(diff_x)
            if diff_x.shape != output.shape:
                output = resize(output, diff_x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            output = output * torch.sigmoid(diff_x) + output

        output = self.cls_seg(output)
        return output