# Copyright (c) Open-CD. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.models.losses import accuracy
from mmseg.utils import ConfigType, SampleList

from opencd.registry import MODELS


class ChangeMixin(BaseModule):
    """This module enables any segmentation model to detect binary change.

    The common usage is to attach this module on a segmentation model without the
    classification head.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2108.07002
    """

    def __init__(
        self,
        in_channels: int = 128 * 2,
        inner_channels: int = 16,
        num_convs: int = 4,
    ):
        """Initializes a new ChangeMixin module.

        Args:
            in_channels: sum of channels of bitemporal feature maps
            inner_channels: number of channels of inner feature maps
            num_convs: number of convolution blocks
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.modules.Sequential(
                nn.modules.Conv2d(in_channels, inner_channels, 3, 1, 1),
                nn.modules.BatchNorm2d(inner_channels),
                nn.modules.ReLU(True),
            )
        ]
        layers += [
            nn.modules.Sequential(
                nn.modules.Conv2d(inner_channels, inner_channels, 3, 1, 1),
                nn.modules.BatchNorm2d(inner_channels),
                nn.modules.ReLU(True),
            )
            for _ in range(num_convs - 1)
        ]

        self.convs = nn.modules.Sequential(*layers)

    def forward(self, x1: Tensor, x2: Tensor) -> List[Tensor]:
        """Forward pass of the model.

        Args:
            x1, x2: input bitemporal feature maps of shape [b, c, h, w]

        Returns:
            a list of bidirected output predictions
        """
        t1t2 = torch.cat([x1, x2], dim=1)
        t2t1 = torch.cat([x2, x1], dim=1)

        c12 = self.convs(t1t2)
        c21 = self.convs(t2t1)

        return [c12, c21]


@MODELS.register_module()
class ChangeStarHead(BaseDecodeHead):
    """The Head of ChangeStar.

    This head is the implementation of
    `ChangeStar <https://arxiv.org/abs/2108.07002>` _.

    Args:
        inference_mode: inference mode of ChangeStar, candidates 
            are `t1t2`, `t2t1`, and `mean`. Default: 't1t2'.
        seg_head_cfg: config for segmentation head.
        changemixin_cfg: config for ChangeMixin.
    """

    def __init__(self, 
                 inference_mode = 't1t2',
                 seg_head_cfg: ConfigType = ...,
                 changemixin_cfg: ConfigType = ...,
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs)
        
        assert kwargs['channels'] == changemixin_cfg['inner_channels'], \
            "`channels` of `ChangeStar` should be same as 'inner_channels`" \
            "in `changemixin_cfg`"

        self.inference_mode = inference_mode

        seg_head_cfg.update(
            num_classes=2,
            out_channels=1,
            dropout_ratio=0.,
            loss_decode=dict(
                type='mmseg.CrossEntropyLoss', use_sigmoid=True, loss_weight=0.)
        ) # useless hyper-parameter

        self.seg_head = MODELS.build(seg_head_cfg)
        self.seg_head.conv_seg = nn.Identity()
        self.changemixin = ChangeMixin(**changemixin_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        x1 = self.seg_head(inputs1)
        x2 = self.seg_head(inputs2)

        c12, c21 = self.changemixin(x1, x2)
        out1 = self.cls_seg(c12)
        out2 = self.cls_seg(c21)
        return out1, out2

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
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
        prefixes = ['t1t2.', 't2t1.']
        for i, seg_logit in enumerate(seg_logits):
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            if self.sampler is not None:
                seg_weight = self.sampler.sample(seg_logit, seg_label)
            else:
                seg_weight = None
            seg_label = seg_label.squeeze(1)

            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    loss[prefixes[i]+loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[prefixes[i]+loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

            loss[prefixes[i]+'acc_seg'] = accuracy(
                seg_logit, seg_label, ignore_index=self.ignore_index)
            seg_label = seg_label.unsqueeze(1)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        if self.inference_mode == 't1t2':
            seg_logits = seg_logits[0]
        elif self.inference_mode == 't2t1':
            seg_logits = seg_logits[1]
        elif self.inference_mode == 'mean':
            seg_logits = (seg_logits[0] + seg_logits[1]) / 2.0
        else:
            raise ValueError(f"Invalid inference_mode: {self.inference_mode}")
        
        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits