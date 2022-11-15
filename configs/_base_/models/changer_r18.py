# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='IA_ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        interaction_cfg=(None, None, None, None)),
    decode_head=dict(
        type='Changer',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))