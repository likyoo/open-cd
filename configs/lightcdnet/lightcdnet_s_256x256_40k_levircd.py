_base_ = [
    '../_base_/models/lightcdnet.py',
    '../common/standard_256x256_40k_levircd.py']

model = dict(
    decode_head=dict(
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.003,
    betas=(0.9, 0.999),
    weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
