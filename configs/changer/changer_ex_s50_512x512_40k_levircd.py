_base_ = [
    '../_base_/models/changer_s50.py', '../_base_/datasets/levir_cd.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
model = dict(
    backbone=dict(
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2))
    ),
    decode_head=dict(
        num_classes=2,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
        # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(type='MultiImgNormalize', **img_norm_cfg),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        img_dir='train',
        ann_dir='train/label',
        pipeline=train_pipeline),
    val=dict(
        img_dir='val',
        ann_dir='val/label',
        pipeline=test_pipeline),
    test=dict(
        img_dir='test',
        ann_dir='test/label',
        pipeline=test_pipeline))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.005,
    betas=(0.9, 0.999),
    weight_decay=0.05)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])
