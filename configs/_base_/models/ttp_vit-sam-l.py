# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
fpn_norm_cfg = dict(type='mmpretrain.LN2d', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

sam_pretrain_ckpt_path = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth'
model = dict(
    type='TimeTravellingPixels',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VisionTransformerTurner',
        encoder_cfg=dict(
            type='ViTSAM_Custom',
            arch='large',
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            layer_cfgs=dict(type='TimeFusionTransformerEncoderLayer'),
            init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path, prefix='backbone.')),
        peft_cfg=dict(
            r=16,
            target_modules=["qkv"],
            lora_dropout=0.01,
            bias='lora_only')),
    neck=dict(
        type='SequentialNeck',
        necks=[
            dict(
                type='FeatureFusionNeck',
                policy='concat',
                out_indices=(0,)),
            dict(
                type='SimpleFPN',
                backbone_channel=512,
                in_channels=[128, 256, 512, 512],
                out_channels=256,
                num_outs=5,
                norm_cfg=fpn_norm_cfg)]),
    decode_head=dict(
        type='MLPSegHead',
        out_size=(128, 128),
        in_channels=[256] * 5,
        in_index=[0, 1, 2, 3, 4],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))