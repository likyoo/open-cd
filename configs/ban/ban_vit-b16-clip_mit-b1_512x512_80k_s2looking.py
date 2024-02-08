_base_ = ['./ban_vit-b16-clip_mit-b0_512x512_80k_s2looking.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

# model settings
model = dict(
    decode_head=dict(
        ban_cfg=dict(
            side_enc_cfg=dict(
                init_cfg=dict(
                    type='Pretrained', checkpoint=checkpoint),
                embed_dims=64)),
        ban_dec_cfg=dict(
            in_channels=[64, 128, 320, 512])))