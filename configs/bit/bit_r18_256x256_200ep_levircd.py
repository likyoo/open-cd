_base_ = [
    '../_base_/models/bit_r18.py', '../_base_/datasets/levir_cd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_200ep.py'
]
model = dict(
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        input_transform='resize_concat',
        in_index=[0,1,2,3],
        in_channels=[64,128,256,512],
        num_classes=2,
        pre_upsample=1),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=2))

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=12,
    )

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.99, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(interval=50, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])