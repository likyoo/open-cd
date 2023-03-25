# Copyright (c) Open-CD. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class MultiImgLoadImageFromFile(object):
    """Load images from files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            if isinstance(results['img_prefix'], list):
                if isinstance( results['img_info']['filename'], list):
                    filenames = [osp.join(ip, fn)
                            for ip, fn in zip(results['img_prefix'], \
                                              results['img_info']['filename'])]
                else:
                    filenames = [osp.join(ip, results['img_info']['filename'])
                                for ip in results['img_prefix']]
            else:
                filenames = [osp.join(results['img_prefix'],
                                     results['img_info']['filename'])]
        else:
            raise ValueError('`img_prefix` == None is not support')
        
        imgs = []
        for filename in filenames:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        results['filename'] = filenames
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape
        results['ori_shape'] = imgs[0].shape
        # Set initial values for default meta_keys
        results['pad_shape'] = imgs[0].shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class MultiImgLoadAnnotations(object):
    """Load annotations for change detection.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`opencd.CDDataset`. 

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='grayscale', # in mmseg: unchanged
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify to format ann
        if results.get('format_ann', None) is not None:
            if results['format_ann'] == 'binary':
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
                gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            else:
                raise ValueError('Invalid value {}'.format(results['format_ann']))
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class MultiImgMultiAnnLoadAnnotations(object):
    """Load annotations for semantic change detection.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_semantic_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_semantic_zero_label = reduce_semantic_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`opencd.CDDataset`. 

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            if isinstance(results['ann_info']['seg_map'], list):
                binary_, semantic_from_, semantic_to_ = results['ann_info']['seg_map']
                binary_filename = osp.join(results['seg_prefix']['binary_dir'], binary_)
                semantic_filename_from = osp.join(results['seg_prefix']['semantic_dir_from'],
                                    semantic_from_)
                semantic_filename_to = osp.join(results['seg_prefix']['semantic_dir_to'],
                                    semantic_to_)
            else:
                binary_filename = osp.join(results['seg_prefix']['binary_dir'],
                                    results['ann_info']['seg_map'])
                semantic_filename_from = osp.join(results['seg_prefix']['semantic_dir_from'],
                                    results['ann_info']['seg_map'])
                semantic_filename_to = osp.join(results['seg_prefix']['semantic_dir_to'],
                                    results['ann_info']['seg_map'])
        else:
            assert NotImplementedError
        # for binary change ann
        binary_img_bytes = self.file_client.get(binary_filename)
        gt_semantic_seg = mmcv.imfrombytes(
            binary_img_bytes, flag='grayscale',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # for semantic anns
        semantic_img_bytes_from = self.file_client.get(semantic_filename_from)
        gt_semantic_seg_from = mmcv.imfrombytes(
            semantic_img_bytes_from, flag='grayscale', # in mmseg: unchanged
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        semantic_img_bytes_to = self.file_client.get(semantic_filename_to)
        gt_semantic_seg_to = mmcv.imfrombytes(
            semantic_img_bytes_to, flag='grayscale', # in mmseg: unchanged
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        
        # modify to format ann
        if results.get('format_ann', None) is not None:
            if results['format_ann'] == 'binary':
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
                gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            else:
                raise ValueError('Invalid value {}'.format(results['format_ann']))
        # modify if custom classes
        if results.get('label_map', None) is not None:
            ''' Just for semantic anns here '''
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_from_copy = gt_semantic_seg_from.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg_from[gt_semantic_seg_from_copy == old_id] = new_id
            gt_semantic_seg_to_copy = gt_semantic_seg_to.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg_to[gt_semantic_seg_to_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_semantic_zero_label:
            ''' Just for semantic anns here '''
            # avoid using underflow conversion
            gt_semantic_seg_from[gt_semantic_seg_from == 0] = 255
            gt_semantic_seg_from = gt_semantic_seg_from - 1
            gt_semantic_seg_from[gt_semantic_seg_from == 254] = 255
            gt_semantic_seg_to[gt_semantic_seg_to == 0] = 255
            gt_semantic_seg_to = gt_semantic_seg_to - 1
            gt_semantic_seg_to[gt_semantic_seg_to == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['gt_semantic_seg_from'] = gt_semantic_seg_from
        results['gt_semantic_seg_to'] = gt_semantic_seg_to
        results['seg_fields'].extend(['gt_semantic_seg', 
            'gt_semantic_seg_from', 'gt_semantic_seg_to'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_semantic_zero_label={self.reduce_semantic_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
