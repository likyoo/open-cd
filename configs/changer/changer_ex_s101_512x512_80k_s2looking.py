_base_ = './changer_ex_s50_512x512_80k_s2looking.py'

model = dict(backbone=dict(depth=101, stem_channels=128))
