_base_ = [
    '../_base_/models/siam_upernet_r18.py', '../_base_/datasets/levir_cd.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        in_channels=[v*2 for v in [64, 128, 256, 512]],
        num_classes=2), 
    auxiliary_head=dict(
        in_channels=256*2,
        num_classes=2))
