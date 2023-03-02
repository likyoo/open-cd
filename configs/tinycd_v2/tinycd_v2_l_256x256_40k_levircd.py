_base_ = ['./tinycd_v2_b_256x256_40k_levircd.py']

model = dict(
    backbone=dict(
        type='TinyNet',
        arch='L',
        stem_stack_nums=4,
        widen_factor=1.2),
    neck=dict(
        type='TinyFPN',
        in_channels=[24, 32, 40, 56],
        out_channels=24,
        num_outs=4),
    decode_head=dict(
        type='TinyHead',
        in_channels=[48, 24, 24, 24, 24],
        channels=24,
        dropout_ratio=0.))