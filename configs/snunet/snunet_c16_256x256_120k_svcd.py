_base_ = [
    '../_base_/models/snunet_c16.py', '../_base_/datasets/svcd.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=120000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 120k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=12000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=12000))