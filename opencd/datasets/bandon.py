# Copyright (c) Open-CD. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets import DATASETS
from mmseg.utils import get_root_logger
from .scd_custom import SCDDataset


@DATASETS.register_module()
class BANDON_Dataset(SCDDataset):
    """BANDON dataset
    
    Note: Use `tools/generate_txt/generate_bandon_txt.py` 
        to generate .txt files for BANDON dataset
    
    """

    SEMANTIC_CLASSES = ('background', 'roofs', 'facades')
    
    SEMANTIC_PALETTE = [[0, 0, 0], [244, 177, 131], [143, 170, 220]]

    def __init__(self, **kwargs):
        super().__init__(
            sub_dir_1='imgs',
            sub_dir_2='imgs',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=('unchanged', 'changed'),
            palette=[[0, 0, 0], [255, 255, 255]],
            reduce_semantic_zero_label=False, # Zero label is needed for evaluation
            inverse_semantic_zero_pred=False,
            **kwargs)
    
    def load_annotations(self, img_dir, img_suffix, sub_dir_1, sub_dir_2,
                         ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip().split(' ')
                # img_name: img1, img2, binary label, semantic_from label, \
                # semantic_to label
                img_info = dict(filename=[name + img_suffix for name in img_name[:2]])
                if ann_dir is not None:
                    seg_map = [name + seg_map_suffix for name in img_name[2:]]
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            raise NotImplementedError(f'Currently only `split` mode is supported, \
                                      You can get .txt split files by running \
                                      `tools/generate_split/generate_bandon_txt.py`')

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

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
        palette = np.array(self.PALETTE) if mode == 'binary' \
            else np.array(self.SEMANTIC_PALETTE)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename_from, filename_to = self.img_infos[idx]['filename']
            city, _, filename = filename_from.split('/')
            t_from, t_to = filename_from.split('/')[1], filename_to.split('/')[1]

            if mode == 'binary':
                imgfile_prefix = osp.join(imgfile_prefix, city, t_from+'VS'+t_to)
            elif mode == 'semantic_from':
                imgfile_prefix = osp.join(imgfile_prefix, city, t_from)
            elif mode == 'semantic_to':
                imgfile_prefix = osp.join(imgfile_prefix, city, t_to)
            else:
                raise ValueError
            mmcv.mkdir_or_exist(imgfile_prefix)

            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
            for idx, color in enumerate(palette):
                color_seg[result == idx, :] = color

            output = Image.fromarray(color_seg.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files
    
    def format_results(self, results, imgfile_prefix, indices=None):
        """Format the results into dir.
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
            imgfile_prefix, indices, mode='binary')
        from_result_files = self.results2img(results[1], \
            imgfile_prefix, indices, mode='semantic_from')
        to_result_files = self.results2img(results[2], \
            imgfile_prefix, indices, mode='semantic_to')

        result_files = np.stack((binary_result_files, \
            from_result_files, to_result_files), axis=1)
        return result_files
