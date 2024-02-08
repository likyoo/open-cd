# Copyright (c) Open-CD. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import caffe2_xavier_init

from mmseg.models.utils import nlc_to_nchw, LayerNorm2d
from mmseg.utils import ConfigType, SampleList
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from opencd.registry import MODELS

from .ban_utils import BridgeLayer


class BitemporalAdapterBranch(nn.Module):
    """The encoder of Bi-temporal Adapter Branch.

    Args:
        clip_channels (int): Number of channels of visual features.
            Default: 768.
        fusion_index (List[int]): The layer number of the encode
            transformer to fuse with the CLIP feature.
            Default: [0, 1, 2].
        side_enc_cfg (ConfigType): Configs for the encode layers.
    """

    def __init__(
            self,
            clip_channels: int = 768,
            fusion_index: list = [0, 1, 2],
            side_enc_cfg: ConfigType = ...,
    ):
        super().__init__()

        self.side_encoder = MODELS.build(side_enc_cfg)
        self.encoder_type = side_enc_cfg.type

        clip_attns = []
        conv_clips = []
        
        if 'MixVisionTransformer' in self.encoder_type:
            side_enc_channels = [num * self.side_encoder.embed_dims
                                    for num in self.side_encoder.num_heads]
            if isinstance(clip_channels, int):
                clip_channels = [clip_channels] * len(side_enc_channels)
            for i in fusion_index:
                conv_clips.append(
                    nn.Sequential(
                        LayerNorm2d(clip_channels[i]),
                        ConvModule(
                            clip_channels[i],
                            side_enc_channels[i],
                            kernel_size=1,
                            norm_cfg=None,
                            act_cfg=None)))
                clip_attns.append(
                    BridgeLayer(
                        num_heads=self.side_encoder.num_heads[i],
                        embed_dims=side_enc_channels[i],
                        kdim=None,
                        vdim=None))
            self.clip_attns = nn.ModuleList(clip_attns)
            self.conv_clips = nn.ModuleList(conv_clips)
        elif 'ResNet' in self.encoder_type:
            side_enc_channels = [self.side_encoder.base_channels * 2**i 
                             for i in [0, 1, 2, 3]]
            num_heads = [1, 2, 4, 8]
            for i in fusion_index:
                conv_clips.append(
                    nn.Sequential(
                        LayerNorm2d(clip_channels),
                        ConvModule(
                            clip_channels,
                            side_enc_channels[i],
                            kernel_size=1,
                            norm_cfg=None,
                            act_cfg=None)))
                clip_attns.append(
                    BridgeLayer(
                        num_heads=num_heads[i],
                        embed_dims=side_enc_channels[i],
                        kdim=None,
                        vdim=None))
            self.clip_attns = nn.ModuleList(clip_attns)
            self.conv_clips = nn.ModuleList(conv_clips)
        else:
            raise NotImplementedError('Do not support encoder '
                                      f'for "{self.encoder_type}"') 

        self.fusion_index = fusion_index

    def init_weights(self):
        self.side_encoder.init_weights()
        for i in range(len(self.conv_clips)):
            caffe2_xavier_init(self.conv_clips[i][1].conv)
        for i in range(len(self.clip_attns)):
            self.clip_attns[i].init_weights()

    def fuse_clip(self, fused_index: int, x: torch.Tensor,
                clip_feature: torch.Tensor):
        """Fuse CLIP feature and visual tokens."""
        clip_fea = self.conv_clips[fused_index](clip_feature.contiguous())
        fused_clip = self.clip_attns[fused_index](x, clip_fea)
        
        return fused_clip

    def encode_feature(self, x: torch.Tensor,
                       clip_features: List[torch.Tensor]) -> List[List]:
        """Encode images by a lightweight vision transformer."""

        outs = []
        fused_index = 0
        cls_token = False
        if isinstance(clip_features[0], list):
            cls_token = True

        if 'MixVisionTransformer' in self.encoder_type:
            for index, layer in enumerate(self.side_encoder.layers, start=0):
                x, hw_shape = layer[0](x)
                for block in layer[1]:
                    x = block(x, hw_shape)
                x = layer[2](x)
                x = nlc_to_nchw(x, hw_shape)

                if index in self.fusion_index:
                    if cls_token:
                        x = self.fuse_clip(fused_index, x,
                                        clip_features[fused_index][0])
                    else:
                        x = self.fuse_clip(fused_index, x,
                                        clip_features[fused_index])
                    fused_index += 1
                outs.append(x)
            return outs
        
        elif 'ResNet' in self.encoder_type:
            if self.side_encoder.deep_stem:
                x = self.side_encoder.stem(x)
            else:
                x = self.side_encoder.conv1(x)
                x = self.side_encoder.norm1(x)
                x = self.side_encoder.relu(x)
            x = self.side_encoder.maxpool(x)
            
            for index, layer_name in enumerate(self.side_encoder.res_layers, start=0):
                res_layer = getattr(self.side_encoder, layer_name)
                x = res_layer(x)

                if index in self.fusion_index:
                    if cls_token:
                        x = self.fuse_clip(fused_index, x,
                                        clip_features[fused_index][0])
                    else:
                        x = self.fuse_clip(fused_index, x,
                                        clip_features[fused_index])
                    fused_index += 1

                if index == self.side_encoder.out_indices[-1]:
                    return x
        else:
            raise NotImplementedError('Do not support encoder '
                                      f'for "{self.encoder_type}"')

    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward function."""
        features = self.encode_feature(image, clip_features)
        return features


@MODELS.register_module()
class BitemporalAdapterHead(BaseDecodeHead):
    """A New Learning Paradigm for Foundation Model-based 
    Remote Sensing Change Detection. `BAN <arxiv.org/abs/2312.01163>` _.

    Args:
        ban_cfg (ConfigType): Configs for BitemporalAdapterBranch
        ban_dec_cfg (ConfigType): Configs for Bi-TAB's decoder
    """

    def __init__(self,
                 ban_cfg: ConfigType,
                 ban_dec_cfg: ConfigType,
                 **kwargs):
        super().__init__(
            in_channels=1,
            channels=1,
            num_classes=ban_dec_cfg.num_classes,
            **kwargs)

        del self.conv_seg

        self.side_adapter_network = BitemporalAdapterBranch(**ban_cfg)
        self.mask_decoder = MODELS.build(ban_dec_cfg)

    def init_weights(self):
        self.side_adapter_network.init_weights()

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[List]:
        """Forward function.

        Args:
            inputs (Tuple[Tensor]): A pair including images,
            list of multi-level visual features from image encoder.

        Returns:
            output (List[Tensor]): Mask predicted by BAN.
        """
        img_from, img_to, fm_feat_from, fm_feat_to = inputs

        mask_props_from = self.side_adapter_network(
            img_from, fm_feat_from)
        
        mask_props_to = self.side_adapter_network(
            img_to, fm_feat_to)
        
        output = self.mask_decoder(mask_props_from, mask_props_to)

        return output

    def predict(self, inputs: Tuple[torch.Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> torch.Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): Images, visual features from image encoder
            and class embedding from text encoder.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        mask_props = self.forward(inputs)

        return self.predict_by_feat(mask_props,
                                    batch_img_metas)

    def loss(self, x: Tuple[torch.Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # forward
        seg_logits = self.forward(x)

        # loss
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        return losses