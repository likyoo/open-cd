# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[122.7709, 116.7460, 104.0937] * 2,
    std=[68.5005, 66.6322, 70.3232] * 2,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='BAN',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/clip_vit-base-patch16-224_3rdparty-d08f8887.pth',
    asymetric_input=True,
    encoder_resolution=dict(
        size=(224, 224),
        mode='bilinear'),
    image_encoder=dict(
        type='mmseg.VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_pad=0,
        in_channels=3,
        embed_dims=768,
        num_layers=9,
        num_heads=12,
        mlp_ratio=4,
        out_origin=False,
        out_indices=(2, 5, 8),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=True,
        patch_bias=False,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        act_cfg=dict(type='mmseg.QuickGELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        frozen_exclude=['pos_embed']),
    decode_head=dict(
        type='BitemporalAdapterHead',
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
