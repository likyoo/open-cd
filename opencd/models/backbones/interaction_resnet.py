# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

from mmseg.models.backbones import ResNet
from mmseg.models.builder import BACKBONES


class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (int, optional): 1/p of the features will be exchanged.
            Defaults to 2.
    """
    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2


class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (int, optional): 1/p of the features will be exchanged.
            Defaults to 2.
    """
    def __init__(self, p=2):
        super(SpatialExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2


class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self, channels, num_paths=2, attn_channels=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(Aggregation_distribution, self).__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


@BACKBONES.register_module()
class IA_ResNet(ResNet):
    """Interaction ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (dict | None): Dictionary to construct and config DCN conv layer.
            When dcn is not None, conv_cfg must be None. Default: None.
        stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
            stage. The length of stage_with_dcn is equal to num_stages.
            Default: (False, False, False, False).
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'.
            Default: None.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from opencd.models import IA_ResNet
        >>> import torch
        >>> self = IA_ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs, inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 128, 8, 8)
        (1, 256, 4, 4)
        (1, 512, 2, 2)
        (1, 1024, 1, 1)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cross-correlation
        self.ccs = []
        for idx in range(self.num_stages):
            if idx in [0, 1]:
                self.ccs.append(SpatialExchange(p=2))
            else:
                self.ccs.append(ChannelExchange(p=2))
        self.ccs = nn.ModuleList(self.ccs)
    
    def forward(self, x1, x2, interact_layer=[1, 2, 3]):
        """Forward function."""
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x
            
        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            if i in interact_layer:
                x1, x2 = self.ccs[i](x1, x2)
            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))
        return tuple(outs)


@BACKBONES.register_module()
class IA_ResNetV1c(IA_ResNet):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super(IA_ResNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class IA_ResNetV1d(IA_ResNet):
    """ResNetV1d variant described in [1]_.
    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(IA_ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)