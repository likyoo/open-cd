_base_ = ['./ttp_vit-sam-l_512x512_300e_levircd.py']

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(
    batch_size=2,
    num_workers=4)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05),
    dtype='float16')