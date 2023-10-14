# Copyright (c) Open-CD. All rights reserved.
import copy
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np

from mmseg.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class BaseSCDDataset(_BaseCDDataset):
    def __init__(self,
                 lazy_init=False,
                 reduce_semantic_zero_label=False,
                 **kwargs):
        super().__init__(lazy_init=True, **kwargs)

        self.reduce_semantic_zero_label = reduce_semantic_zero_label

        # Get label map for semantic custom classes
        new_classes = self._metainfo.get('semantic_classes', None)
        self.semantic_label_map = self.get_semantic_label_map(new_classes)
        self._metainfo.update(
            dict(
                semantic_label_map=self.semantic_label_map,
                reduce_semantic_zero_label=self.reduce_semantic_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_semantic_palette = self._update_semantic_palette()
        self._metainfo.update(dict(semantic_palette=updated_semantic_palette))

        if not lazy_init:
            self.full_init()

        if self.test_mode:
            assert self._metainfo.get('semantic_classes') is not None, \
                'dataset metainfo `semantic_classes` should be specified when testing'

    @classmethod
    def get_semantic_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require semantic label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get('semantic_classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['semantic_classes']):
                raise ValueError(
                    f'new semantic_classes {new_classes} is not a '
                    f'subset of semantic_classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_semantic_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('semantic_palette', [])
        classes = self._metainfo.get('semantic_classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.semantic_label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.semantic_label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette


    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir_from = self.data_prefix.get('img_path_from', None)
        img_dir_to = self.data_prefix.get('img_path_to', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        ann_dir_from = self.data_prefix.get('seg_map_path_from', None)
        ann_dir_to = self.data_prefix.get('seg_map_path_to', None)

        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                data_names = line.strip().split(' ')
                # img_name: img1, img2, binary label, semantic_from label, \
                # semantic_to label
                img_name_from, img_name_to, ann_name, ann_name_from, \
                                                        ann_name_to = data_names

                data_info = dict(img_path=\
                                 [osp.join(img_dir_from, img_name_from + self.img_suffix), \
                                  osp.join(img_dir_to, img_name_to + self.img_suffix)])
                if ann_dir is not None:
                    seg_map = ann_name + self.seg_map_suffix
                    seg_map_from = ann_name_from + self.seg_map_suffix
                    seg_map_to = ann_name_to + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['seg_map_path_from'] = osp.join(ann_dir_from, seg_map_from)
                    data_info['seg_map_path_to'] = osp.join(ann_dir_to, seg_map_to)
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['semantic_label_map'] = self.semantic_label_map
                data_info['reduce_semantic_zero_label'] = self.reduce_semantic_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            file_list_from = fileio.list_dir_or_file(
                    dir_path=img_dir_from,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args)
            file_list_to = fileio.list_dir_or_file(
                    dir_path=img_dir_to,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args)

            assert sorted(list(file_list_from)) == sorted(list(file_list_to)), \
                'The images in `img_path_from` and `img_path_to` are not ' \
                    'one-to-one correspondence'

            for img in fileio.list_dir_or_file(
                    dir_path=img_dir_from,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=\
                                 [osp.join(img_dir_from, img), \
                                  osp.join(img_dir_to, img)])
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['seg_map_path_from'] = osp.join(ann_dir_from, seg_map)
                    data_info['seg_map_path_to'] = osp.join(ann_dir_to, seg_map)
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['semantic_label_map'] = self.semantic_label_map
                data_info['reduce_semantic_zero_label'] = self.reduce_semantic_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
