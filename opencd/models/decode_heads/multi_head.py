# Copyright (c) Open-CD. All rights reserved.

from abc import ABCMeta, abstractmethod

import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import HEADS


@HEADS.register_module()
class MultiHeadDecoder(BaseModule):
    """Base class for MultiHeadDecoder.

    Args:
        binary_cd_head (dict): The decode head for binary change detection branch.
        binary_cd_neck (dict): The feature fusion part for binary \
            change detection branch
        semantic_cd_head (dict): The decode head for semantic change \
            detection `from` branch.
        semantic_cd_head_aux (dict): The decode head for semantic change \
            detection `to` branch. If None, the siamese semantic head will \
            be used. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 binary_cd_head,
                 binary_cd_neck,
                 semantic_cd_head=None,
                 semantic_cd_head_aux=None,
                 init_cfg=None):
        super(MultiHeadDecoder, self).__init__(init_cfg)
        self.binary_cd_head = builder.build_head(binary_cd_head)
        self.binary_cd_neck = builder.build_neck(binary_cd_neck)
        self.siamese_semantic_head = True
        if semantic_cd_head is not None:
            self.semantic_cd_head = builder.build_head(semantic_cd_head)
            if semantic_cd_head_aux is not None:
                self.siamese_semantic_head = False
                self.semantic_cd_head_aux = builder.build_head(semantic_cd_head_aux)
            else:
                self.semantic_cd_head_aux = self.semantic_cd_head

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function.
        The return value should be a dict() containing: 
        `binary_cd_logit`, `semantic_cd_logit_from` and 
        `semantic_cd_logit_to`.
        
        For example:
            return dict(
                binary_cd_logit=out,
                semantic_cd_logit_from=out1, 
                semantic_cd_logit_to=out2)
        """
        pass

    def forward_train(self, inputs, img_metas, gt_dict, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        assert isinstance(seg_logits, dict), "`seg_logits` should be a dict()"
        assert ['binary_cd_logit', 'semantic_cd_logit_from', 'semantic_cd_logit_to'] \
            == list(seg_logits.keys()), "`binary_cd_logit`, `semantic_cd_logit_from` \
            and `semantic_cd_logit_to` should be contained."
        
        losses = self.losses(seg_logits, gt_dict)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs)
        assert isinstance(seg_logits, dict), "`seg_logits` should be a dict()"
        assert ['binary_cd_logit', 'semantic_cd_logit_from', 'semantic_cd_logit_to'] \
            == list(seg_logits.keys()), "`binary_cd_logit`, `semantic_cd_logit_from` \
            and `semantic_cd_logit_to` should be contained."
        return seg_logits

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        losses = dict()
        binary_cd_loss_decode = self.binary_cd_head.losses(
            seg_logit['binary_cd_logit'], seg_label['binary_cd_gt'])
        losses.update(add_prefix(binary_cd_loss_decode, 'binary_cd'))

        if getattr(self, 'semantic_cd_head'):
            semantic_cd_loss_decode_from = self.semantic_cd_head.losses(
                seg_logit['semantic_cd_logit_from'], seg_label['semantic_cd_gt_from'])
            losses.update(add_prefix(semantic_cd_loss_decode_from, 'semantic_cd_from'))

            semantic_cd_loss_decode_to = self.semantic_cd_head_aux.losses(
                seg_logit['semantic_cd_logit_to'], seg_label['semantic_cd_gt_to'])
            losses.update(add_prefix(semantic_cd_loss_decode_to, 'semantic_cd_to'))

        return losses