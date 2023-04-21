# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basescddataset import BaseSCDDataset


@DATASETS.register_module()
class SECOND_Dataset(BaseSCDDataset):
    """SECOND dataset"""
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]],
        semantic_classes=('unchanged', 'water', 'ground', 
                          'low vegetation', 'tree', 'building',
                          'sports field'),
        semantic_palette=[[255, 255, 255], [0, 0, 255], [128, 128, 128], 
                          [0, 128, 0], [0, 255, 0], [128, 0, 0], 
                          [255, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_semantic_zero_label=True,
                 **kwargs) -> None:
        
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_semantic_zero_label=reduce_semantic_zero_label,
            **kwargs)