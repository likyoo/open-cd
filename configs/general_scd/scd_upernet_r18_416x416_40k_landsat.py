_base_ = [
    '../_base_/models/scd_upernet_r18.py', '../_base_/datasets/landsat.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(postprocess_pred_and_label=None)

optimizer=dict(
    type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)

default_hooks = dict(visualization=dict(type='CDVisualizationHook', \
                                        interval=1, draw_on_from_to_img=False))

visualizer = dict(type='CDLocalVisualizer', alpha=1.0)

# compile = True