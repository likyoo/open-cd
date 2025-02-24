from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from .siamencoder_decoder import SiamEncoderDecoder
from .dual_input_encoder_decoder import DIEncoderDecoder
from .ban import BAN
from .ttp import TimeTravellingPixels


@MODELS.register_module()
class DistillSiamEncoderDecoder(SiamEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        self.teacher_l = SiamEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_m = SiamEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_m,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_s = SiamEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
        
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        self.teacher_l.eval()
        self.teacher_m.eval()
        self.teacher_s.eval()
        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例
            
            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward(self.teacher_m.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses


@MODELS.register_module()
class DistillSiamEncoderDecoder_ChangeStar(SiamEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        self.teacher_l = SiamEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_m = SiamEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_m,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_s = SiamEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
        
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs_1 = []
        teacher_outputs_2 = []

        self.teacher_l.eval()
        self.teacher_m.eval()
        self.teacher_s.eval()
        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例
            
            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward(self.teacher_m.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs_1.append(teacher_output[0])
            teacher_outputs_2.append(teacher_output[1])

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs_1 = torch.cat(teacher_outputs_1, dim=0)
        teacher_outputs_2 = torch.cat(teacher_outputs_2, dim=0)

        teacher_outputs = (teacher_outputs_1, teacher_outputs_2)
        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses

    
@MODELS.register_module()
class DistillBAN(BAN):
    def __init__(self,
                 distill_loss,
                 image_encoder: ConfigType,
                 decode_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 asymetric_input: bool = True,
                 encoder_resolution: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        super().__init__(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg)

        self.teacher_l = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_l)
        
        self.teacher_m = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_m)
        
        self.teacher_s = BAN(
                 image_encoder=image_encoder,
                 decode_head=decode_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 data_preprocessor=data_preprocessor,
                 pretrained=pretrained,
                 asymetric_input=asymetric_input,
                 encoder_resolution=encoder_resolution,
                 init_cfg=init_cfg_t_s)
        
        self.distill_loss = MODELS.build(distill_loss)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        img_from, img_to = torch.split(inputs, 3, dim=1)

        fm_img_from, fm_img_to = img_from, img_to
        if self.asymetric_input:
            fm_img_from = F.interpolate(
                fm_img_from, **self.encoder_resolution)
            fm_img_to = F.interpolate(
                fm_img_to, **self.encoder_resolution)
        fm_feat_from = self.image_encoder(fm_img_from)
        fm_feat_to = self.image_encoder(fm_img_to)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            [img_from, img_to, fm_feat_from, fm_feat_to], data_samples)
        losses.update(loss_decode)
        

        student_output = self.decode_head.forward([img_from, img_to, fm_feat_from, fm_feat_to])  # 学生模型的输出
        
        teacher_outputs = []
        
        self.teacher_s.eval()
        self.teacher_m.eval()
        self.teacher_l.eval()
        
        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_s.extract_feat(img_from[i:i+1]), self.teacher_s.extract_feat(img_to[i:i+1])])
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_m.extract_feat(img_from[i:i+1]), self.teacher_m.extract_feat(img_to[i:i+1])])
                else:
                    teacher_output = self.teacher_l.decode_head.forward([img_from[i:i+1], img_to[i:i+1], self.teacher_l.extract_feat(img_from[i:i+1]), self.teacher_l.extract_feat(img_to[i:i+1])])

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses


@MODELS.register_module()
class DistillTimeTravellingPixels(TimeTravellingPixels):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        self.teacher_l = TimeTravellingPixels(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_m = TimeTravellingPixels(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_m,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_s = TimeTravellingPixels(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
        
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward(self.teacher_m.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses


@MODELS.register_module()
class DistillDIEncoderDecoder_S(DIEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        self.teacher_l = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_m = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_m,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_s = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
            
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward(self.teacher_m.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses