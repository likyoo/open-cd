# Copyright (c) Open-CD. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets import DATASETS
from .scd_custom import SCDDataset


@DATASETS.register_module()
class Landsat_Dataset(SCDDataset):
    """Landsat dataset"""

    SEMANTIC_CLASSES = ('unchanged', 'farmland', 'desert', 
                        'building', 'water')
    
    SEMANTIC_PALETTE = [[255, 255, 255], [128, 128, 128], [130, 87, 87], 
                        [255, 0, 0], [0, 0, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            sub_dir_1='im1',
            sub_dir_2='im2',
            img_suffix='.png',
            seg_map_suffix='.png',
            classes=('unchanged', 'changed'),
            palette=[[0, 0, 0], [255, 255, 255]],
            reduce_semantic_zero_label=False, # Zero label is needed for evaluation
            inverse_semantic_zero_pred=True, # Inverse zero pred is needed for evaluation
            **kwargs)

    def results2img(self, results, imgfile_prefix, indices=None, mode='binary'):
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
        palette = np.array(self.SEMANTIC_PALETTE)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            if mode == 'semantic':
                color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
                for idx, color in enumerate(palette):
                    color_seg[result == idx, :] = color

            elif mode == 'binary':
                color_seg = result * 255 # for binary change detection
            output = Image.fromarray(color_seg.astype(np.uint8))
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

        if self.inverse_semantic_zero_pred:
            results = self.inverse_reduce_zero_label(results)
        results = [list(pred) for pred in list(zip(*results))]
        binary_result_files = self.results2img(results[0], \
            osp.join(imgfile_prefix, 'binary'), indices, mode='binary')
        from_result_files = self.results2img(results[1], \
            osp.join(imgfile_prefix, 'from'), indices, mode='semantic')
        to_result_files = self.results2img(results[2], \
            osp.join(imgfile_prefix, 'to'), indices, mode='semantic')

        result_files = np.stack((binary_result_files, \
            from_result_files, to_result_files), axis=1)
        return result_files
