from .bcl_loss import BCLLoss
from .kd_loss import DistillLoss, DistillLossWithPixel, DistillLossWithPixel_ChangeStar

__all__ = ['BCLLoss', 'DistillLoss', 'DistillLossWithPixel', 'DistillLossWithPixel_ChangeStar']