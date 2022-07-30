import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16


@NECKS.register_module()
class FeatureFusionNeck(BaseModule):
    """Feature Fusion Neck.

    This neck is the implementation of
    `SFNet <https://arxiv.org/abs/2002.10120>`_.

    Args:
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3)):
        super(FeatureFusionNeck, self).__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.fp16_enabled = False

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'Lp_distance':
            x = torch.abs(x1 - x2)

        return x

    @auto_fp16()
    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)