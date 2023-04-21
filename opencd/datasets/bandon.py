# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basescddataset import BaseSCDDataset


@DATASETS.register_module()
class BANDON_Dataset(BaseSCDDataset):
    """BANDON dataset
    
    Note: Use `tools/generate_txt/generate_bandon_txt.py` 
        to generate .txt files for BANDON dataset
    
    """
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]],
        semantic_classes=('background', 'roofs', 'facades'),
        semantic_palette=[[0, 0, 0], [244, 177, 131], [143, 170, 220]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_semantic_zero_label=False,
                 **kwargs) -> None:
        
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_semantic_zero_label=reduce_semantic_zero_label,
            **kwargs)