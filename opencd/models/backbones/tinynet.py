# Copyright (c) Open-CD. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils import checkpoint as cp

from mmseg.models.utils import SELayer, make_divisible
from opencd.registry import MODELS


class AsymGlobalAttn(BaseModule):
    def __init__(self, dim, strip_kernel_size=21):
        super().__init__()

        self.norm = build_norm_layer(dict(type='mmpretrain.LN2d', eps=1e-6), dim)[1]
        self.global_ = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim, (1, strip_kernel_size), padding=(0, (strip_kernel_size-1)//2), groups=dim),
                nn.Conv2d(dim, dim, (strip_kernel_size, 1), padding=((strip_kernel_size-1)//2, 0), groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        a = self.global_(x)
        x = a * self.v(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x + identity

        return x


class PriorAttention(BaseModule):
    def __init__(self, 
                 channels, 
                 num_paths=2, 
                 attn_channels=None, 
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(PriorAttention, self).__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        attn = x.mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)

        return x1 * attn1 + x1, x2 * attn2 + x2


class StemBlock(BaseModule):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 **kwargs):
        super(StemBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs),
        ])
        
        self.conv = nn.Sequential(*layers)
        self.interact = PriorAttention(channels=hidden_dim)
        self.post_conv = ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                **kwargs)

    def forward(self, x):
        x1, x2 = x
        identity_x1 = x1
        identity_x2 = x2
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1, x2 = self.interact(x1, x2)
        x1 = self.post_conv(x1)
        x2 = self.post_conv(x2)

        if self.use_res_connect:
            x1 = x1 + identity_x1
            x2 = x2 + identity_x2

        return x1, x2


class PriorFusion(BaseModule):
    def __init__(self, channels, stack_nums=2):
        super().__init__()

        self.stem = nn.Sequential(
            *[StemBlock(
                in_channels=channels,
                out_channels=channels,
                stride=1,
                expand_ratio=4) for _ in range(stack_nums)])

        self.pseudo_fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
                build_norm_layer(dict(type='mmpretrain.LN2d', eps=1e-6), channels * 2)[1],
                nn.GELU(),
                nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels),
        )


    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        identity_x1 = x1
        identity_x2 = x2
        
        x1, x2 = self.stem((x1, x2))
        x1 = x1 + identity_x1
        x2 = x2 + identity_x2

        early_x = torch.cat([x1, x2], dim=1)
        x = self.pseudo_fusion(early_x)
        return early_x, x


class TinyBlock(BaseModule):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 with_se=False,
                 **kwargs):
        super(TinyBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        Attention_Layer = SELayer(hidden_dim) if with_se else nn.Identity()
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs),
            Attention_Layer,
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                **kwargs)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                x = x + self.conv(x)
                return x
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@MODELS.register_module()
class TinyNet(BaseModule):
    """TinyNet backbone.
    This backbone is the implementation of

    Args:
        output_early_x (bool): output early features before fusion.
            Defaults to 'False'.
        arch='B' (str): The model's architecture. It should be
            one of architecture in ``TinyNet.change_extractor_settings``.
            Defaults to 'B'.
        stem_stack_nums (int): The number of stacked stem blocks.
        use_global: (Sequence[bool]): whether use `AsymGlobalAttn` after 
            stages. Defaults: (True, True, True, True).
        strip_kernel_size: (Sequence[int]): The strip kernel size of 
            `AsymGlobalAttn`. Defaults: (41, 31, 21, 11).
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        strides (Sequence[int], optional): Strides of the first block of each
            layer. If not specified, default config in ``arch_setting`` will
            be used.
        dilations (Sequence[int]): Dilation of each layer.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    change_extractor_settings = {
        'S': [[4, 16, 2], [6, 24, 2], [6, 32, 3], [6, 48, 1]],
        'B': [[4, 16, 2], [6, 24, 2], [6, 32, 3], [6, 48, 1]], 
        'L': [[4, 16, 2], [6, 24, 2], [6, 32, 6], [6, 48, 1]],}

    def __init__(self,
                 output_early_x=False,
                 arch='B',
                 stem_stack_nums=2,
                 use_global=(True, True, True, True),
                 strip_kernel_size=(41, 31, 21, 11),
                 widen_factor=1.,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.arch_settings = self.change_extractor_settings[arch]
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 7). But received {index}')

        if frozen_stages not in range(-1, 7):
            raise ValueError('frozen_stages must be in range(-1, 7). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(16 * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fusion_block = PriorFusion(self.in_channels, stem_stack_nums)

        self.layers = []
        self.use_global = use_global
        self.strip_kernel_size = strip_kernel_size

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio,
                use_global=use_global[i],
                strip_kernel_size=self.strip_kernel_size[i])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        
        self.output_early_x = output_early_x

    def make_layer(self, out_channels, num_blocks, stride, dilation,
                   expand_ratio, use_global, strip_kernel_size):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.
        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                TinyBlock(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels
        # after stage
        if use_global:
            layers.append(
                AsymGlobalAttn(out_channels, strip_kernel_size)
            )

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        
        early_x, x = self.fusion_block(x1, x2)

        if self.output_early_x:
            outs = [early_x]
        else:
            outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(TinyNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()