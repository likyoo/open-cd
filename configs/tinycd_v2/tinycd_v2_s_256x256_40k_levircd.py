_base_ = ['./tinycd_v2_b_256x256_40k_levircd.py']

model = dict(
    backbone=dict(
        type='TinyNet',
        arch='S',
        stem_stack_nums=2,
        widen_factor=0.5),
    neck=dict(
        type='TinyFPN',
        in_channels=[8, 16, 16, 24],
        out_channels=24,
        num_outs=4),
    decode_head=dict(
        type='TinyHead',
        in_channels=[16, 24, 24, 24, 24],
        channels=24,
        dropout_ratio=0.))