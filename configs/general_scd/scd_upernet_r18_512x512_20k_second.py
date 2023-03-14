_base_ = [
    '../_base_/models/scd_upernet_r18.py', '../_base_/datasets/second.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.05)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

evaluation = dict(_delete_=True, interval=2000, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='semantic.mSCD_Score', greater_keys=['mSCD_Score'])

