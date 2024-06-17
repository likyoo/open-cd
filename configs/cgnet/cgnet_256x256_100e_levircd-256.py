_base_ = [
    '../_base_/models/cgnet.py', 
    '../common/standard_256x256_100e_levircd-256.py']

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0025)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
