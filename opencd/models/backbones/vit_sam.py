# Copyright (c) Open-CD. All rights reserved.
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmpretrain.models.utils import LayerNorm2d, resize_pos_embed, to_2tuple
from mmpretrain.models.backbones.vit_sam import TransformerEncoderLayer
from mmpretrain.models.backbones.base_backbone import BaseBackbone

from opencd.registry import MODELS


@MODELS.register_module()
class ViTSAM_Custom(BaseBackbone):
    """Vision Transformer as image encoder used in SAM.

    A PyTorch implement of backbone: `Segment Anything
    <https://arxiv.org/abs/2304.02643>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'base', 'large', 'huge'. If use dict, it should have
            below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            - **global_attn_indexes** (int): The index of layers with global
              attention.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_channels (int): The num of output channels, if equal to 0, the
            channel reduction layer is disabled. Defaults to 256.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        out_type (str): The type of output features. Please choose from

            - ``"raw"`` or ``"featmap"``: The feature map tensor from the
              patch tokens with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).

            Defaults to ``"raw"``.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        use_abs_pos (bool): Whether to use absolute position embedding.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to True.
        window_size (int): Window size for window attention. Defaults to 14.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072,
                'global_attn_indexes': [2, 5, 8, 11]
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096,
                'global_attn_indexes': [5, 11, 17, 23]
            }),
        **dict.fromkeys(
            ['h', 'huge'], {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120,
                'global_attn_indexes': [7, 15, 23, 31]
            }),
    }
    OUT_TYPES = {'raw', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 out_indices: int = -1,
                 out_type: str = 'raw',
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = True,
                 window_size: int = 14,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 frozen_stages: int = -1,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.global_attn_indexes = self.arch_settings['global_attn_indexes']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.use_abs_pos = use_abs_pos
        self.interpolate_mode = interpolate_mode
        if use_abs_pos:
            # Set position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, *self.patch_resolution, self.embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)
            self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        if use_rel_pos:
            self._register_load_state_dict_pre_hook(
                self._prepare_relative_position)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                window_size=window_size
                if i not in self.global_attn_indexes else 0,
                input_size=self.patch_resolution,
                use_rel_pos=use_rel_pos,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            if 'type' in _layer_cfg:
                self.layers.append(MODELS.build(_layer_cfg))
            else:
                self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.out_channels = out_channels
        if self.out_channels > 0:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
            )

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # freeze channel_reduction module
        if self.frozen_stages == self.num_layers and self.out_channels > 0:
            m = self.channel_reduction
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        x = x.view(B, patch_resolution[0], patch_resolution[1],
                   self.embed_dims)

        if self.use_abs_pos:
            # 'resize_pos_embed' only supports 'pos_embed' with ndim==3, but
            # in ViTSAM, the 'pos_embed' has 4 dimensions (1, H, W, C), so it
            # is flattened. Besides, ViTSAM doesn't have any extra token.
            resized_pos_embed = resize_pos_embed(
                self.pos_embed.flatten(1, 2),
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0)
            x = x + resized_pos_embed.view(1, *patch_resolution,
                                           self.embed_dims)
            x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self.out_indices:
                # (B, H, W, C) -> (B, C, H, W)
                x_reshape = x.permute(0, 3, 1, 2)

                if self.out_channels > 0:
                    x_reshape = self.channel_reduction(x_reshape)
                outs.append(self._format_output(x_reshape))

        return tuple(outs)

    def _format_output(self, x) -> torch.Tensor:
        if self.out_type == 'raw' or self.out_type == 'featmap':
            return x
        elif self.out_type == 'avg_featmap':
            # (B, C, H, W) -> (B, C, N) -> (B, N, C)
            x = x.flatten(2).permute(0, 2, 1)
            return x.mean(dim=1)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = ckpt_pos_embed_shape[1:3]
            pos_embed_shape = self.patch_embed.init_out_size

            flattened_pos_embed = state_dict[name].flatten(1, 2)
            resized_pos_embed = resize_pos_embed(flattened_pos_embed,
                                                 ckpt_pos_embed_shape,
                                                 pos_embed_shape,
                                                 self.interpolate_mode, 0)
            state_dict[name] = resized_pos_embed.view(1, *pos_embed_shape,
                                                      self.embed_dims)

    def _prepare_relative_position(self, state_dict, prefix, *args, **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'rel_pos_' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_pretrained = state_dict[ckpt_key]
                relative_position_current = state_dict_model[key]
                L1, _ = relative_position_pretrained.size()
                L2, _ = relative_position_current.size()
                if L1 != L2:
                    new_rel_pos = F.interpolate(
                        relative_position_pretrained.reshape(1, L1,
                                                             -1).permute(
                                                                 0, 2, 1),
                        size=L2,
                        mode='linear',
                    )
                    new_rel_pos = new_rel_pos.reshape(-1, L2).permute(1, 0)
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(f'Resize the {ckpt_key} from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos.shape}')
                    state_dict[ckpt_key] = new_rel_pos

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers