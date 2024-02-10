# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.ops import CrissCrossAttention

from mmseg.models.utils import LayerNorm2d
from opencd.registry import MODELS


class CCA(nn.Module):
    """Criss-Cross Attention for Semantic Segmentation.

    This head is the implementation of `CCNet
    <https://arxiv.org/abs/1811.11721>`_.

    Args:
        recurrence (int): Number of recurrence of Criss Cross Attention
            module. Default: 2.
    """

    def __init__(self, channels, recurrence=2):
        super(CCA, self).__init__()
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(channels)

    def forward(self, x):
        for _ in range(self.recurrence):
            x = self.cca(x)
        return x


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class ShuffleBlock(nn.Module):

    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                # 3*3 dw conv, stride = 2
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 2
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

        else:
            assert in_c == out_c

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

    def forward(self, x):
        out = None
        if self.downsample:
            # if it is downsampling, we don't need to do channel split
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)

        return channel_shuffle(out, 2)


class TimeAttention(nn.Module):

    def __init__(self, channels):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        attn_channels = channels // 16
        attn_channels = max(attn_channels, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, channels * 2, kernel_size=1, bias=False),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.avg_pool(x)
        y = self.mlp(x)
        B, C, H, W = y.size()
        x1_attn, x2_attn = y.reshape(B, 2, C // 2, H, W).transpose(0, 1)
        x1_attn = torch.sigmoid(x1_attn)
        x2_attn = torch.sigmoid(x2_attn)
        x1 = x1 * x1_attn + x1
        x2 = x2 * x2_attn + x2
        return x1, x2


class shuffle_fusion(nn.Module):

    def __init__(self, channels, block_num=2):
        super().__init__()

        self.stages = []
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels * 4), nn.ReLU()))
        for i in range(block_num):
            self.stages.append(
                ShuffleBlock(channels * 4, channels * 4, downsample=False))

        self.stages = nn.Sequential(*self.stages)

        self.single_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())

        self.time_attn = TimeAttention(channels)

        self.final_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())

    def forward_single(self, x):
        identity = x
        x = self.stages(x)
        x = self.single_conv(x)
        x = identity + x
        return x

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x1, x2 = self.time_attn(x1, x2)
        x = self.final_conv(channel_shuffle(torch.cat((x1, x2), dim=1)))
        return x


@MODELS.register_module()
class LightCDNet(nn.Module):

    def __init__(self, stage_repeat_num, net_type="small"):
        super(LightCDNet, self).__init__()

        index_list = stage_repeat_num.copy()
        index_list[0] = index_list[0] - 1
        self.index_list = list(np.cumsum(index_list))
        if net_type == "small":
            self.out_channels = [24, 48, 96, 192]
            self.block_num = 4
        elif net_type == "base":
            self.out_channels = [24, 116, 232, 464]
            self.block_num = 8
        elif net_type == "large":
            self.out_channels = [24, 176, 352, 704]
            self.block_num = 16
        else:
            print("the model type is error!")

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.out_channels[0], 3, 2, 1, bias=False),
            LayerNorm2d(self.out_channels[0]), nn.GELU())

        self.fusion_conv = shuffle_fusion(
            self.out_channels[0], block_num=self.block_num)

        in_c = self.out_channels[0]

        self.stages = []
        for stage_idx in range(len(stage_repeat_num)):
            out_c = self.out_channels[1 + stage_idx]
            repeat_num = stage_repeat_num[stage_idx]
            for i in range(repeat_num):
                if i == 0:
                    self.stages.append(
                        ShuffleBlock(in_c, out_c, downsample=True))
                else:
                    self.stages.append(
                        ShuffleBlock(in_c, in_c, downsample=False))
                in_c = out_c
            self.stages.append(CCA(channels=out_c, recurrence=2))

        self.stages = nn.Sequential(*self.stages)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x = self.fusion_conv(x1, x2)
        outs = [x]

        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i in self.index_list:
                outs.append(x)
        return outs
