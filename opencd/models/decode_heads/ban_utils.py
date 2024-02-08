import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import (ConvModule, build_norm_layer, Conv2d, 
                      ConvModule, build_activation_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, Sequential, ModuleList

from mmseg.models.utils import Upsample
from mmseg.models.utils import nlc_to_nchw, nchw_to_nlc
from mmseg.models.utils import resize

from opencd.registry import MODELS
from opencd.models.decode_heads.bit_head import TransformerEncoder, TransformerDecoder


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(self,
                 dim: int,
                 layer_scale_init_value: float = 1e-5,
                 inplace: bool = False,
                 data_format: str = 'channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight


class CrossMultiheadAttention(MultiheadAttention):
    """
    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 layer_scale_init_value=0):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            kdim=kdim,
            vdim=vdim,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)
        
        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value, \
                    data_format='channels_first')
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x_q, x_kv, identity=None):

        if identity is None:
            identity = x_q
        
        hw_shape = x_q.shape[-2:]
        x_q = nchw_to_nlc(x_q)
        x_kv = nchw_to_nlc(x_kv)

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)
        
        out = nlc_to_nchw(out, hw_shape)

        return identity + self.gamma1(out)


class BridgeLayer(BaseModule):
    """Bridging Modele in BAN.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 feedforward_channels=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='mmpretrain.LN2d'),
                 batch_first=True):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = CrossMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, x_kv):
        hwshape = x.shape[-2:]
        x = self.attn(self.norm1(x), x_kv, identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        x = x + resize(
            x_kv,
            size=hwshape,
            mode='bilinear',
            align_corners=False)
        return x
    

class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head of Changer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


@MODELS.register_module()
class BAN_MLPDecoder(BaseModule):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 norm_cfg=None,
                 dropout_ratio=0.1,
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 interpolate_mode='bilinear'):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        self.out_channels = num_classes

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.discriminator = MixFFN(
            embed_dims=self.channels * 2,
            feedforward_channels=self.channels * 2,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))
        
        self.conv_seg = nn.Conv2d(self.channels * 2, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
      
    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        
        return out

    def forward(self, inputs1, inputs2):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = torch.cat([out1, out2], dim=1)

        out = self.discriminator(out)
        out = self.cls_seg(out)

        return out


@MODELS.register_module()
class BAN_BITHead(BaseModule):
    """BIT Head

    This head is the improved implementation of'Remote Sensing Image
    Change Detection With Transformers<https://github.com/justchenhao/BIT_CD>'

    Args:
        in_channels (int): Number of input feature channels (from backbone). Default:  512
        channels (int): Number of output channels of pre_process. Default:  32.
        embed_dims (int): Number of expanded channels of Attention block. Default:  64.
        enc_depth (int): Depth of block of transformer encoder. Default:  1.
        enc_with_pos (bool): Using position embedding in transformer encoder.
            Default:  True
        dec_depth (int): Depth of block of transformer decoder. Default:  8.
        num_heads (int): Number of Multi-Head Cross-Attention Head of transformer encoder.
            Default:  8.
        use_tokenizer (bool),Using semantic token. Default:  True
        token_len (int): Number of dims of token. Default:  4.
        pre_upsample (int): Scale factor of upsample of pre_process.
            (default upsample to 64x64)
            Default: 2.
    """

    def __init__(self,
                 in_channels=256,
                 channels=32,
                 num_classes=2,
                 embed_dims=64,
                 enc_depth=1,
                 enc_with_pos=True,
                 dec_depth=8,
                 num_heads=8,
                 drop_rate=0.,
                 pool_size=2,
                 pool_mode='max',
                 use_tokenizer=True,
                 token_len=4,
                 pre_upsample=2,
                 upsample_size=4,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 align_corners=False,
                 interpolate_mode='bilinear',
                 conv_cfg=None,
                 dropout_ratio=0.):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.interpolate_mode = interpolate_mode
        self.out_channels = num_classes
        self.conv_cfg = conv_cfg

        self.embed_dims=embed_dims
        self.use_tokenizer = use_tokenizer
        self.num_heads=num_heads
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.token_len = token_len
            self.conv_att = ConvModule(
                self.channels,
                self.token_len,
                1,
                conv_cfg=self.conv_cfg,
            )

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, self.channels))

        # pre_process to backbone feature
        self.pre_process = Sequential(
            Upsample(scale_factor=pre_upsample, mode='bilinear', align_corners=self.align_corners),
            ConvModule(
                self.in_channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg
            )
        )

        # Transformer Encoder
        self.encoder = ModuleList()
        for _ in range(enc_depth):
            block = TransformerEncoder(
                self.channels,
                self.embed_dims,
                self.num_heads,
                drop_rate=drop_rate,
                norm_cfg=self.norm_cfg,
            )
            self.encoder.append(block)

        # Transformer Decoder
        self.decoder = ModuleList()
        for _ in range(dec_depth):
            block = TransformerDecoder(
                self.channels,
                self.embed_dims,
                self.num_heads,
                drop_rate=drop_rate,
                norm_cfg=self.norm_cfg,
            )
            self.decoder.append(block)

        self.upsample = Upsample(scale_factor=upsample_size,mode='bilinear',align_corners=self.align_corners)

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
    
    # Token
    def _forward_semantic_tokens(self, x):
        b, c = x.shape[:2]
        att_map = self.conv_att(x)
        att_map = att_map.reshape((b, self.token_len, 1, -1)).contiguous()
        att_map = F.softmax(att_map, dim=-1)
        x = x.reshape((b, 1, c, -1)).contiguous()
        tokens = (x * att_map).sum(-1)
        return tokens

    def _forward_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.permute((0, 2, 3, 1)).flatten(1, 2)
        return tokens

    def _forward_feature(self, x1, x2):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x1 = self.pre_process(x1)
        x2 = self.pre_process(x2)
        # Tokenization
        if self.use_tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshaped_tokens(x1)
            token2 = self._forward_reshaped_tokens(x2)

        # Transformer encoder forward
        token = torch.cat([token1, token2], dim=1)
        if self.enc_with_pos:
            token += self.enc_pos_embedding
        for i, _encoder in enumerate(self.encoder):
            token = _encoder(token)
        token1, token2 = torch.chunk(token, 2, dim=1)

        # Transformer decoder forward
        for _decoder in self.decoder:
            b, c, h, w = x1.shape
            x1 = x1.permute((0, 2, 3, 1)).flatten(1, 2).contiguous()
            x2 = x2.permute((0, 2, 3, 1)).flatten(1, 2).contiguous()
            
            x1 = _decoder(x1, token1)
            x2 = _decoder(x2, token2)
            
            x1 = x1.transpose(1, 2).reshape((b, c, h, w)).contiguous()
            x2 = x2.transpose(1, 2).reshape((b, c, h, w)).contiguous()

        # Feature differencing
        y = torch.abs(x1 - x2)
        y = self.upsample(y)

        return y

    def forward(self, x1, x2):
        """Forward function."""
        output = self._forward_feature(x1, x2)
        output = self.cls_seg(output)
        return output