# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class RSIPAC_CD_Dataset(_BaseCDDataset):
    """RSIPAC_CD dataset"""
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
