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
model = dict(
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='LightCDNet',
        stage_repeat_num=[4, 8, 4],
        net_type="small"),
    neck=dict(
        type='TinyFPN',
        exist_early_x=True,
        early_x_for_fpn=True,
        custom_block='conv',
        in_channels=[24, 48, 96, 192],
        out_channels=48,
        num_outs=4),
    decode_head=dict(
        type='DS_FPNHead',
        in_channels=[48, 48, 48, 48],
        in_index=[0, 1, 2, 3],
        channels=48,
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=24,
        in_index=0,
        channels=24,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))