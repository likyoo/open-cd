# Copyright (c) Open-CD. All rights reserved.
from mmseg.models.builder import HEADS
from .multi_head import MultiHeadDecoder


@HEADS.register_module()
class GeneralSCDHead(MultiHeadDecoder):
    """The Head of General Semantic Change Detection Head."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        inputs1, inputs2 = inputs
        out1 = self.semantic_cd_head(inputs1)
        out2 = self.semantic_cd_head_aux(inputs2)
        inputs_ = self.binary_cd_neck(inputs1, inputs2)
        out = self.binary_cd_head(inputs_)

        out_dict = dict(
            binary_cd_logit=out,
            semantic_cd_logit_from=out1, 
            semantic_cd_logit_to=out2
        )

        return out_dict