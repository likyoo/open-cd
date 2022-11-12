_base_ = 'snunet_c16_256x256_40k_levircd.py'

base_channels = 32
model = dict(
    backbone=dict(base_channel=base_channels),
    decode_head=dict(
        in_channels=base_channels * 4,
        channels=base_channels * 4))