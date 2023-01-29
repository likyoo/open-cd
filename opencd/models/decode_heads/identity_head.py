# Copyright (c) OpenCD. All rights reserved.
import torch
import torch.nn as nn

from mmcv.runner import force_fp32

from mmseg.ops import resize
from mmseg.models.losses import accuracy
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    """Identity Head."""

    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(channels=1, **kwargs)
        self.conv_seg = nn.Identity()

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output


@HEADS.register_module()
class DSIdentityHead(BaseDecodeHead):
    """Deep Supervision Identity Head."""

    def __init__(self, **kwargs):
        super(DSIdentityHead, self).__init__(channels=1, **kwargs)
        self.conv_seg = nn.Identity()

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_label_size = seg_label.shape[2:]
        for seg_idx, single_seg_logit in enumerate(seg_logit):
            single_seg_logit = resize(
                input=single_seg_logit,
                size=seg_label_size,
                mode='bilinear',
                align_corners=self.align_corners)
            if self.sampler is not None:
                seg_weight = self.sampler.sample(single_seg_logit, seg_label)
            else:
                seg_weight = None
            seg_label = seg_label.squeeze(1)

            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            for loss_decode in losses_decode:
                loss_name = f'aux_{seg_idx}_' + loss_decode.loss_name
                if loss_decode.loss_name not in loss:
                    loss[loss_name] = loss_decode(
                        single_seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_name] += loss_decode(
                        single_seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            single_seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss