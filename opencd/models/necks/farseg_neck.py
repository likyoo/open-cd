# Copyright (c) Open-CD. All rights reserved.
import torch.nn.functional as F

from mmseg.models.necks import FPN
from opencd.registry import MODELS

from .feature_fusion import FeatureFusionNeck


@MODELS.register_module()
class FarSegFPN(FPN):
    def __init__(self, policy='concat', **kwargs):
        super().__init__(**kwargs)
        self.feature_fusion = \
            FeatureFusionNeck(policy, out_indices=tuple(
                range(self.num_outs + 1)))
    
    def base_forward(self, inputs):
        fpn_feats = super().forward(inputs)

        # Extract scene embedding from the last feature map of backbone
        coarsest_features = inputs[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)

        outs = fpn_feats + (scene_embedding,)
        return tuple(outs)
    
    def forward(self, x1, x2):
        out1 = self.base_forward(x1)
        out2 = self.base_forward(x2)
        out = self.feature_fusion(out1, out2)
        return out