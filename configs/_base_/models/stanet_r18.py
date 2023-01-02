# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SiamEncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        type='STAHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=96,
        sa_mode='PAM',
        sa_in_channels=256,
        sa_ds=1,
        distance_threshold=1,
        out_channels=1,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='BCLLoss', margin=2.0, loss_weight=1.0, ignore_index=255)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))