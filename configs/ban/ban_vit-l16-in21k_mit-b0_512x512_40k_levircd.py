_base_ = ['ban_vit-b16-in21k_mit-b0_512x512_40k_levircd.py']

vit_checkpoint_file = 'pretrain/augreg_L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.pth'  # noqa

model = dict(
    pretrained=None,
    asymetric_input=True,
    encoder_resolution=dict(
        size=(224, 224),
        mode='bilinear'),
    image_encoder=dict(
        type='mmseg.VisionTransformer',
        init_cfg=dict(
            type='Pretrained', checkpoint=vit_checkpoint_file),
        img_size=(224, 224),
        patch_size=16,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(11, 17, 23)),
    decode_head=dict(
        type='BitemporalAdapterHead',
        ban_cfg=dict(
            clip_channels=1024,
            fusion_index=[1, 2, 3]),
        ban_dec_cfg=dict(
            in_channels=[32, 64, 160, 256],
            channels=128,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)))

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=1)
