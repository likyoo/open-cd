# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SiamEncoderMultiDecoder',
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
    decode_head=dict(
        type='GeneralSCDHead',
        binary_cd_neck=dict(type='FeatureFusionNeck', policy='concat'),
        binary_cd_head=dict(
            type='UPerHead',
            in_channels=[v * 2 for v in [64, 128, 256, 512]],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=128,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        semantic_cd_head=dict(
            type='UPerHead',
            in_channels=[64, 128, 256, 512],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=128,
            dropout_ratio=0.1,
            num_classes=6,
            ignore_index=255,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True))),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))