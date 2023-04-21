_base_ = [
    '../_base_/models/stanet_r18.py',
    '../common/standard_256x256_40k_levircd.py']

crop_size = (256, 256)
model = dict(
    decode_head=dict(sa_mode='None'),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )
