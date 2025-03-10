_base_ = ['./mtkd-changeformer_mit-b0_512x512_200k_jl1cd.py']

checkpoint_student = None
checkpoint_teacher_l = None
checkpoint_teacher_m = None
checkpoint_teacher_s = None

# model settings
model = dict(
    # student
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher medium    
    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),
    
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    
    decode_head=dict(in_channels=[v * 2 for v in [64, 128, 320, 512]]))