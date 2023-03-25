# dataset settings
dataset_type = 'BANDON_Dataset'
data_root = 'data/BANDON'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgMultiAnnLoadAnnotations'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 
        'gt_semantic_seg_from', 'gt_semantic_seg_to']),
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(2048, 2048),
        # img_ratios=[0.75, 1.0, 1.25],
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir=dict(
            binary_dir='train/labels_unch0ch1ig255',
            semantic_dir_from='train/building_labels',
            semantic_dir_to='train/building_labels'),
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir=dict(
            binary_dir='val/labels',
            semantic_dir_from='val/building_labels',
            semantic_dir_to='val/building_labels'),
        split='val.txt',
        format_ann = 'binary',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir=dict(
            binary_dir='val/labels',
            semantic_dir_from='val/building_labels',
            semantic_dir_to='val/building_labels'),
        split='val.txt',
        format_ann = 'binary',
        pipeline=test_pipeline))