# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(interval=20, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])