# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from opencd.registry import MODELS


def bcl_loss(
    pred, 
    target, 
    margin=2.0, 
    eps=1e-4, 
    ignore_index=255, 
    **kwargs):
    pred = pred.squeeze()
    target = target.squeeze()
    assert pred.size() == target.size() and target.numel() > 0
    mask = (target != ignore_index).float()
    target = target * mask
    utarget = 1 - target
    n_u = utarget.sum() + eps
    n_c = target.sum() + eps
    loss = torch.sum(utarget * torch.pow(pred, 2) * mask) / n_u + \
        torch.sum(target * torch.pow(torch.clamp(margin - pred, min=0.), 2)) / n_c
    return loss


@MODELS.register_module()
class BCLLoss(nn.Module):
    """Batch-balanced Contrastive Loss"""

    def __init__(
        self, 
        margin=2.0, 
        loss_weight=1.0, 
        ignore_index=255,
        loss_name='bcl_loss',
        **kwargs):
        super().__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                **kwargs):

        loss = self.loss_weight * bcl_loss(
            pred, target, self.margin, self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name