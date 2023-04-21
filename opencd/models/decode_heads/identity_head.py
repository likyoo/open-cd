# Copyright (c) OpenCD. All rights reserved.
import torch
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from opencd.registry import MODELS


@MODELS.register_module()
class IdentityHead(BaseDecodeHead):
    """Identity Head."""

    def __init__(self, **kwargs):
        super().__init__(channels=1, **kwargs)
        delattr(self, 'conv_seg')
    
    def init_weights(self):
        pass

    def _forward_feature(self, inputs):
        """
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


@MODELS.register_module()
class DSIdentityHead(BaseDecodeHead):
    """Deep Supervision Identity Head."""

    def __init__(self, **kwargs):
        super().__init__(channels=1, **kwargs)
        delattr(self, 'conv_seg')
    
    def init_weights(self):
        pass

    def _forward_feature(self, inputs):
        """
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

    def loss_by_feat(self, seg_logits, batch_data_samples):
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_label_size = seg_label.shape[2:]
        for seg_idx, single_seg_logit in enumerate(seg_logits):
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
