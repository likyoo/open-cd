# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class DSIFN_Dataset(_BaseCDDataset):
    """DSIFN dataset"""
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, **kwargs)
