_base_ = [
    '../_base_/models/changestar_farseg_1x96_r18.py', 
    '../common/standard_512x512_40k_levircd.py']

# optimizer
optimizer=dict(
    type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)