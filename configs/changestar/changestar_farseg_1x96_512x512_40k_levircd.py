_base_ = [
    '../_base_/models/changestar_farseg_1x96_r18.py', 
    '../common/standard_512x512_40k_levircd.py']


train_dataloader = dict(batch_size=16, num_workers=8)

optimizer=dict(
    type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)