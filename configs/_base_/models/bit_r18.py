# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
bit_norm_cfg = dict(type='LN', requires_grad=True)
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
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='mmseg.ResNetV1c',
        depth=18,
        num_stages=3,
        out_indices=(2,),
        dilations=(1, 1, 1),
        strides=(1, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FeatureFusionNeck', 
        policy='concat',
        out_indices=(0,)),
    decode_head=dict(
        type='BITHead',
        in_channels=256,
        channels=32,
        embed_dims=64,
        enc_depth=1,
        enc_with_pos=True,
        dec_depth=8,
        num_heads=8,
        drop_rate=0.,
        use_tokenizer=True,
        token_len=4,
        upsample_size=4,
        num_classes=2,
        norm_cfg=bit_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))