# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class LEVIR_CD_Dataset(_BaseCDDataset):
    """LEVIR-CD dataset"""
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 format_seg_map='to_binary',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)
