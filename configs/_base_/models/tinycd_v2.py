# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TinyNet',
        arch='B',
        output_early_x=True,
        use_global=(True, True, True, True),
        strip_kernel_size=(41, 31, 21, 11),
        widen_factor=1.0,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU6'),
        norm_eval=False,
        with_cp=False,
        pretrained=None,
        init_cfg=None),
    neck=dict(
        type='TinyFPN',
        exist_early_x=True,
        in_channels=[16, 24, 32, 48],
        out_channels=24,
        num_outs=4),
    decode_head=dict(
        type='TinyHead',
        priori_attn=True,
        in_channels=[32, 24, 24, 24, 24],
        in_index=[0, 1, 2, 3, 4],
        feature_strides=[2, 2, 4, 8, 16],
        channels=24,
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))