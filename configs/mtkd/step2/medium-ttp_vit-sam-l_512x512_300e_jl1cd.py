_base_ = [
    '../../_base_/models/ttp_vit-sam-l.py', 
    '../../common/standard_512x512_300e_jl1cd.py']

dataset_type = 'LEVIR_CD_Dataset'
data_root = 'data/JL1-CD'

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

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='train_m/label',
            img_path_from='train_m/A', 
            img_path_to='train_m/B'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val_m/label',
            img_path_from='val_m/A',
            img_path_to='val_m/B'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='test_m/label',
            img_path_from='test_m/A',
            img_path_to='test_m/B'),
        pipeline=test_pipeline))

# optimizer
max_epochs = 200

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