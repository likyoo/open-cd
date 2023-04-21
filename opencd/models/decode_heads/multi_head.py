# Copyright (c) Open-CD. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from mmengine.model import BaseModule
from mmengine.structures import PixelData
from torch import Tensor, nn

# from mmseg.models import builder
from mmseg.models.utils import resize
from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, add_prefix
from opencd.registry import MODELS


@MODELS.register_module()
class MultiHeadDecoder(BaseModule):
    """Base class for MultiHeadDecoder.

    Args:
        binary_cd_head (dict): The decode head for binary change detection branch.
        binary_cd_neck (dict): The feature fusion part for binary \
            change detection branch
        semantic_cd_head (dict): The decode head for semantic change \
            detection `from` branch.
        semantic_cd_head_aux (dict): The decode head for semantic change \
            detection `to` branch. If None, the siamese semantic head will \
            be used. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 binary_cd_head,
                 binary_cd_neck=None,
                 semantic_cd_head=None,
                 semantic_cd_head_aux=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.binary_cd_head = MODELS.build(binary_cd_head)
        self.siamese_semantic_head = True
        if binary_cd_neck is not None:
            self.binary_cd_neck = MODELS.build(binary_cd_neck)
        if semantic_cd_head is not None:
            self.semantic_cd_head = MODELS.build(semantic_cd_head)
            if semantic_cd_head_aux is not None:
                self.siamese_semantic_head = False
                self.semantic_cd_head_aux = MODELS.build(semantic_cd_head_aux)
            else:
                self.semantic_cd_head_aux = self.semantic_cd_head

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function.
        The return value should be a dict() containing: 
        `seg_logits`, `seg_logits_from` and `seg_logits_to`.
        
        For example:
            return dict(
                seg_logits=out,
                seg_logits_from=out1, 
                seg_logits_to=out2)
        """
        pass

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    
    def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
                **kwargs) -> List[Tensor]:
        """Forward function for testing."""
        seg_logits = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas, **kwargs)
        
    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        assert ['seg_logits', 'seg_logits_from', 'seg_logits_to'] \
            == list(seg_logits.keys()), "`seg_logits`, `seg_logits_from` \
            and `seg_logits_to` should be contained."

        self.align_corners = {
            'seg_logits': self.binary_cd_head.align_corners,
            'seg_logits_from': self.semantic_cd_head.align_corners,
            'seg_logits_to': self.semantic_cd_head_aux.align_corners}

        for seg_name, seg_logit in seg_logits.items():
            seg_logits[seg_name] = resize(
                input=seg_logit,
                size=batch_img_metas[0]['img_shape'],
                mode='bilinear',
                align_corners=self.align_corners[seg_name])
        return seg_logits
    
    def get_sub_batch_data_samples(self, batch_data_samples: SampleList, 
                                   sub_metainfo_name: str,
                                   sub_data_name: str) -> list:
        sub_batch_sample_list = []
        for i in range(len(batch_data_samples)):
            data_sample = SegDataSample()

            gt_sem_seg_data = dict(
                data=batch_data_samples[i].get(sub_data_name).data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

            img_meta = {}
            seg_map_path = batch_data_samples[i].metainfo.get(sub_metainfo_name)
            for key in batch_data_samples[i].metainfo.keys():
                if not 'seg_map_path' in key:
                    img_meta[key] = batch_data_samples[i].metainfo.get(key)
            img_meta['seg_map_path'] = seg_map_path
            data_sample.set_metainfo(img_meta)

            sub_batch_sample_list.append(data_sample)
        return sub_batch_sample_list

    def loss_by_feat(self, seg_logits: dict,
                     batch_data_samples: SampleList, **kwargs) -> dict:
        """Compute segmentation loss."""
        assert ['seg_logits', 'seg_logits_from', 'seg_logits_to'] \
            == list(seg_logits.keys()), "`seg_logits`, `seg_logits_from` \
            and `seg_logits_to` should be contained."

        losses = dict()
        binary_cd_loss_decode = self.binary_cd_head.loss_by_feat(
            seg_logits['seg_logits'],
            self.get_sub_batch_data_samples(batch_data_samples,
                                            sub_metainfo_name='seg_map_path',
                                            sub_data_name='gt_sem_seg'))
        losses.update(add_prefix(binary_cd_loss_decode, 'binary_cd'))

        if getattr(self, 'semantic_cd_head'):
            semantic_cd_loss_decode_from = self.semantic_cd_head.loss_by_feat(
                seg_logits['seg_logits_from'],
                self.get_sub_batch_data_samples(batch_data_samples,
                                                sub_metainfo_name='seg_map_path_from',
                                                sub_data_name='gt_sem_seg_from'))
            losses.update(add_prefix(semantic_cd_loss_decode_from, 'semantic_cd_from'))

            semantic_cd_loss_decode_to = self.semantic_cd_head_aux.loss_by_feat(
                seg_logits['seg_logits_to'],
                self.get_sub_batch_data_samples(batch_data_samples,
                                                sub_metainfo_name='seg_map_path_to',
                                                sub_data_name='gt_sem_seg_to'))
            losses.update(add_prefix(semantic_cd_loss_decode_to, 'semantic_cd_to'))

        return losses