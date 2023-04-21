_base_ = [
    '../_base_/models/scd_upernet_r18.py', '../_base_/datasets/second.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(postprocess_pred_and_label='cover_semantic')

# optimizer
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=10000,
        eta_min=1e-5,
        by_epoch=False,
    )
]

optimizer=dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=10.)}))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1000),
                     visualization=dict(type='CDVisualizationHook', \
                                        interval=1, draw_on_from_to_img=False))

visualizer = dict(type='CDLocalVisualizer', alpha=1.0)

# compile = True