# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

# distill_loss 配置
distill_loss = dict(
    type='DistillLossWithPixel',
    temperature=2.0,     
    loss_weight=0.0,
    pixel_weight=0.001,
)

base_channels = 16

model = dict(
    type='DistillDIEncoderDecoder_S',
    distill_loss=distill_loss,  # 根据需求传入的 distill_loss 配置
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SNUNet_ECAM',
        in_channels=3,
        base_channel=base_channels),
    decode_head=dict(
        type='mmseg.FCNHead',
        in_channels=base_channels * 4,
        channels=base_channels * 4,
        in_index=-1,
        num_convs=0,
        concat_input=False,
        num_classes=2,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))