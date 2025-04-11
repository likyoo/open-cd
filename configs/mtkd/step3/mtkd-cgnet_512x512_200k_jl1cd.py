_base_ = [
    '../../_base_/models/mtkd/mtkd-cgnet.py', 
    '../../common/standard_512x512_200k_jl1cd.py']

crop_size = (512, 512)

checkpoint_student = None
checkpoint_teacher_l = None
checkpoint_teacher_m = None
checkpoint_teacher_s = None

model = dict(
    # student
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_student),
    # teacher large    
    init_cfg_t_l = dict(type='Pretrained', checkpoint=checkpoint_teacher_l),
    # teacher medium    
    init_cfg_t_m = dict(type='Pretrained', checkpoint=checkpoint_teacher_m),
    # teacher small    
    init_cfg_t_s = dict(type='Pretrained', checkpoint=checkpoint_teacher_s),
    
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
)

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