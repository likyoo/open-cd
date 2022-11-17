# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TinyCD',
        in_channels=3,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=8,
        channels=8,
        in_index=-1,
        num_convs=0,
        concat_input=False,
        num_classes=2,
        out_channels=1, # support single class
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))