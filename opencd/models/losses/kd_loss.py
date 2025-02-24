# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from opencd.registry import MODELS


def kd_loss(
    pred_s, 
    pred_t,
    eps=1e-4, 
    ignore_index=255, 
    **kwargs):
    assert pred_s.size() == pred_t.size()    
    mask = (pred_t != ignore_index).float()
    n_pixel = mask.sum() + eps
    loss = torch.sum(torch.pow(pred_s-pred_t, 2)) / n_pixel
    return loss


@MODELS.register_module()
class DistillLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, 
                 temperature=1.0,         # 温度参数，用于调节教师模型输出的平滑度
                 loss_weight=1.0,         # 蒸馏损失权重
                 loss_name='distill_loss', # 损失项名称
                 **kwargs):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, 
                student_pred, 
                teacher_pred, 
                **kwargs):
        """
        计算学生模型和教师模型输出之间的KL散度蒸馏损失。
        
        Args:
            student_pred (Tensor): 学生模型的预测输出 (N, C, H, W)。
            teacher_pred (Tensor): 教师模型的预测输出 (N, C, H, W)。

        Returns:
            Tensor: 计算得到的蒸馏损失。
        """
        # 将学生和教师模型的输出进行 softmax 归一化
        student_logits = nn.functional.log_softmax(student_pred / self.temperature, dim=1)
        teacher_logits = nn.functional.softmax(teacher_pred / self.temperature, dim=1)

        # 使用 KL 散度计算学生与教师的输出差异
        distill_loss = nn.functional.kl_div(
            student_logits, teacher_logits, reduction='batchmean') * (self.temperature ** 2)

        # 加权损失
        num_elements = student_logits.numel()
        # loss = self.loss_weight / num_elements * distill_loss
        loss = self.loss_weight * distill_loss
        return loss

    @property
    def loss_name(self):
        """返回损失项的名称。"""
        return self._loss_name
    
    
@MODELS.register_module()
# 新的蒸馏损失模块
class DistillLossWithPixel(nn.Module):
    """改进版知识蒸馏损失，包含像素级和类别级蒸馏。"""

    def __init__(self, temperature=1.0, loss_weight=1.0, pixel_weight=1.0, loss_name='distill_loss', **kwargs):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self._loss_name = loss_name

    def forward(self, student_pred, teacher_pred, **kwargs):
        # 类别级蒸馏
        student_logits = nn.functional.log_softmax(student_pred / self.temperature, dim=1)
        teacher_logits = nn.functional.softmax(teacher_pred / self.temperature, dim=1)
        class_distill_loss = nn.functional.kl_div(
            student_logits, teacher_logits, reduction='batchmean') * (self.temperature ** 2)

        # 像素级蒸馏
        pixel_distill_loss = nn.functional.mse_loss(student_pred, teacher_pred)

        # 综合蒸馏损失
        total_loss = (self.loss_weight * class_distill_loss +
                      self.pixel_weight * pixel_distill_loss)
        return total_loss

    @property
    def loss_name(self):
        """返回损失项的名称。"""
        return self._loss_name


@MODELS.register_module()
# 新的蒸馏损失模块
class DistillLossWithPixel_ChangeStar(nn.Module):
    """改进版知识蒸馏损失，包含像素级和类别级蒸馏。"""

    def __init__(self, temperature=1.0, loss_weight=1.0, pixel_weight=1.0, loss_name='distill_loss', **kwargs):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.pixel_weight = pixel_weight
        self._loss_name = loss_name

    def forward(self, student_pred, teacher_pred, **kwargs):
        student_pred = torch.cat([student_pred[0], student_pred[1]], dim=0)
        teacher_pred = torch.cat([teacher_pred[0], teacher_pred[1]], dim=0)
        # 类别级蒸馏
        student_logits = nn.functional.log_softmax(student_pred / self.temperature, dim=1)
        teacher_logits = nn.functional.softmax(teacher_pred / self.temperature, dim=1)
        class_distill_loss = nn.functional.kl_div(
            student_logits, teacher_logits, reduction='batchmean') * (self.temperature ** 2)

        # 像素级蒸馏
        pixel_distill_loss = nn.functional.mse_loss(student_pred, teacher_pred)

        # 综合蒸馏损失
        total_loss = (self.loss_weight * class_distill_loss +
                      self.pixel_weight * pixel_distill_loss)
        return total_loss

    @property
    def loss_name(self):
        """返回损失项的名称。"""
        return self._loss_name
