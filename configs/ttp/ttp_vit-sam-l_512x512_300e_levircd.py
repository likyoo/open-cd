_base_ = [
    '../_base_/models/ttp_vit-sam-l.py', 
    '../common/standard_256x256_100e_levircd.py']

crop_size = (512, 512)

model = dict(
    backbone=dict(
        encoder_cfg=dict(img_size=crop_size)),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)))

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(pipeline=train_pipeline))

# optimizer
max_epochs = 300

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-4, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        begin=5,
        by_epoch=True,
        end=max_epochs,
        convert_to_iter_based=True
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
default_hooks = dict(checkpoint=dict(interval=5))