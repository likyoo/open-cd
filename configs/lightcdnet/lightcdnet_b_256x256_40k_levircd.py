_base_ = ['./lightcdnet_s_256x256_40k_levircd.py']

model = dict(
    backbone=dict(net_type="base"),
    neck=dict(in_channels=[24, 116, 232, 464]))