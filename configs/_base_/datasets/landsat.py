# dataset settings
dataset_type = 'Landsat_Dataset'
data_root = 'data/Landsat'

crop_size = (416, 416)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgMultiAnnLoadAnnotations'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5),
    dict(type='MultiImgPackSegInputs')
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(416, 416), keep_ratio=True),
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
        data_prefix=dict(
            img_path_from='train/im1',
            img_path_to='train/im2',
            seg_map_path='train/label',
            seg_map_path_from='train/label1',
            seg_map_path_to='train/label2'),
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
            img_path_from='val/im1',
            img_path_to='val/im2',
            seg_map_path='val/label',
            seg_map_path_from='val/label1',
            seg_map_path_to='val/label2'),
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
            img_path_from='test/im1',
            img_path_to='test/im2',
            seg_map_path='test/label',
            seg_map_path_from='test/label1',
            seg_map_path_to='test/label2'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='SCDMetric', 
    iou_metrics=['mFscore', 'mIoU'], 
    cal_sek=True)
test_evaluator = val_evaluator
