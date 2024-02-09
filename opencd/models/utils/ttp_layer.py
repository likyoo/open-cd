# Copyright (c) Open-CD. All rights reserved.
from typing import Tuple, Optional

import einops
import torch
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import FFN

from mmpretrain.models import build_norm_layer
from mmpretrain.models.backbones.vit_sam import Attention, window_partition, window_unpartition

from opencd.registry import MODELS


@MODELS.register_module()
class TimeFusionTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if self.window_size > 0: # TODO: Maybe it should be ``self.window_size == 0`` here.
            in_channels = embed_dims * 2
            self.down_channel = torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel.weight.data.fill_(1.0 / in_channels)

            self.soft_ffn = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        # time phase fusion
        if self.window_size > 0: # TODO: Maybe it should be ``self.window_size == 0`` here.
            x = einops.rearrange(x, 'b h w d -> b d h w')  # 2B, C, H, W
            x0 = x[:x.size(0)//2]
            x1 = x[x.size(0)//2:]  # B, C, H, W
            x0_1 = torch.cat([x0, x1], dim=1)
            activate_map = self.down_channel(x0_1)
            activate_map = torch.sigmoid(activate_map)
            x0 = x0 + self.soft_ffn(x1 * activate_map)
            x1 = x1 + self.soft_ffn(x0 * activate_map)
            x = torch.cat([x0, x1], dim=0)
            x = einops.rearrange(x, 'b d h w -> b h w d')
        return x
