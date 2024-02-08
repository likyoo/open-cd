# Copyright (c) Open-CD. All rights reserved.
from typing import List

import torch
from torch import Tensor

from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class TimeTravellingPixels(SiamEncoderDecoder):
    """Time Travelling Pixels.
    TimeTravellingPixels typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        img = torch.cat([img_from, img_to], dim=0)
        img_feat = self.backbone(img)[0]
        feat_from, feat_to = torch.split(img_feat, img_feat.shape[0] // 2, dim=0)
        feat_from = [feat_from]
        feat_to = [feat_to]
        if self.with_neck:
            x = self.neck(feat_from, feat_from)
        else:
            raise ValueError('`NECK` is needed for `TimeTravellingPixels`.')

        return x