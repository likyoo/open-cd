# dataset settings
dataset_type = 'BANDON_Dataset'
data_root = 'data/BANDON'

crop_size = (512, 512)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgMultiAnnLoadAnnotations'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5),
    dict(type='MultiImgPackSegInputs')
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(2048, 2048), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgMultiAnnLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(
            img_path_from='train/imgs',
            img_path_to='train/imgs',
            seg_map_path='train/labels_unch0ch1ig255',
            seg_map_path_from='train/building_labels',
            seg_map_path_to='train/building_labels'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        format_seg_map='to_binary',
        data_prefix=dict(
            img_path_from='val/imgs',
            img_path_to='val/imgs',
            seg_map_path='val/labels',
            seg_map_path_from='val/building_labels',
            seg_map_path_to='val/building_labels'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.txt',
        format_seg_map='to_binary',
        data_prefix=dict(
            img_path_from='val/imgs',
            img_path_to='val/imgs',
            seg_map_path='val/labels',
            seg_map_path_from='val/building_labels',
            seg_map_path_to='val/building_labels'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='SCDMetric', 
    iou_metrics=['mFscore', 'mIoU'], 
    cal_sek=True)
test_evaluator = val_evaluator
