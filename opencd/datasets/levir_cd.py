# Copyright (c) Open-CD. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets import DATASETS
from .custom import CDDataset

@DATASETS.register_module()
class LEVIR_CD_Dataset(CDDataset):
    """LEVIR-CD dataset"""

    def __init__(self, **kwargs):
        super().__init__(
            sub_dir_1='A',
            sub_dir_2='B',
            img_suffix='.png',
            seg_map_suffix='.png',
            classes=('unchanged', 'changed'),
            palette=[[255, 255, 255], [0, 0, 0]],
            **kwargs)
        
        self.format_ann = 'binary'

    def results2img(self, results, imgfile_prefix, indices=None):
        """Write the segmentation results to images.
        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 6.
            result = result * 255 # for binary change detection
            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files
    
    def format_results(self, results, imgfile_prefix, indices=None):
        """Format the results into dir (standard format for LoveDA evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, indices)

        return result_files
