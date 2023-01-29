# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='IFN'),
    auxiliary_head=dict(
        type='DSIdentityHead',
        in_channels=[1, 1, 1, 1],
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        num_classes=2,
        out_channels=1, # support single class
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    decode_head=dict(
        type='IdentityHead',
        in_channels=1,
        in_index=-1,
        num_classes=2,
        out_channels=1, # support single class
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))