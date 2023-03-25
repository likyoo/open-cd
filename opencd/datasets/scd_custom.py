# Copyright (c) Open-CD. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import (add_prefix, eval_metrics, intersect_and_union,
                        pre_eval_to_metrics)
from mmseg.datasets import DATASETS
from mmseg.datasets.pipelines import Compose
from opencd.datasets.pipelines import MultiImgMultiAnnLoadAnnotations
from .custom import CDDataset


@DATASETS.register_module()
class SCDDataset(CDDataset):
    """Custom datasets for Semantic change detection. An example of file 
    structure is as followed.
    .. code-block:: none
        ├── data
        │   ├── my_dataset
        │   │   ├── train
        │   │   │   ├── img1_dir/xxx{img_suffix}
        │   │   │   ├── img2_dir/xxx{img_suffix}
        │   │   │   ├── binary_label_dir/xxx{img_suffix}
        │   │   │   ├── semantic_label_dir1/xxx{img_suffix}
        │   │   │   ├── semantic_label_dir2/xxx{img_suffix}
        │   │   ├── val
        │   │   │   ├── img1_dir/xxx{seg_map_suffix}
        │   │   │   ├── img2_dir/xxx{seg_map_suffix}
        │   │   │   ├── binary_label_dir/xxx{seg_map_suffix}
        │   │   │   ├── semantic_label_dir1/xxx{seg_map_suffix}
        │   │   │   ├── semantic_label_dir2/xxx{seg_map_suffix}

    The imgs/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        sub_dir_1 (str): Path to the directory of the first temporal images.
            e.g. 'im1' in SECOND dataset (SECOND/train/im1). Default: 'im1'
        sub_dir_2 (str): Path to the directory of the second temporal images.
            e.g. 'im2' in SECOND dataset (SECOND/train/im2). Default: 'im2'
        img_suffix (str): Suffix of images. Default: '.png'
        ann_dir (dict): Path to annotation directory. 
            Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        inverse_semantic_zero_pred (bool): Whether inverse zero pred in
            semantic anns for evaluation and visualization. Default: False
        reduce_zero_label (bool): Whether to mark label zero as ignored in 
            binary anns. Default: False
        reduce_semantic_zero_label (bool): Whether to mark label zero as 
            ignored in semantic anns. Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        format_ann (str): If `format_ann`='binary', the binary change detection
            label will be formatted as 0 (<128) or 1 (>=128). Default: None
        gt_seg_map_loader_cfg (dict, optional): build MultiImgLoadAnnotations 
            to load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = None
    SEMANTIC_CLASSES = None

    PALETTE = None
    SEMANTIC_PALETTE = None

    def __init__(
        self,
        pipeline,
        img_dir,
        sub_dir_1='im1',
        sub_dir_2='im2',
        img_suffix='.png',
        ann_dir=None,
        seg_map_suffix='.png',
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        inverse_semantic_zero_pred=False,
        reduce_zero_label=False,
        reduce_semantic_zero_label=False,
        classes=None,
        palette=None,
        semantic_classes=None,
        semantic_palette=None,
        format_ann=None,
        gt_seg_map_loader_cfg=None,
        file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.sub_dir_1 = sub_dir_1
        self.sub_dir_2 = sub_dir_2
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.inverse_semantic_zero_pred = inverse_semantic_zero_pred
        self.reduce_zero_label = reduce_zero_label
        self.reduce_semantic_zero_label = reduce_semantic_zero_label
        self.label_map = None
        # In some change detection datasets, the label values
        # are not strictly 0/255, which may cause "RuntimeError:
        # CUDA error: an illegal memory access was encountered".
        # The `format_ann='binary'` will take effect when
        # building `MultiImgLoadAnnotations` PIPELINES.
        self.format_ann = format_ann
        self.CLASSES, self.PALETTE = self.get_binary_classes_and_palette(
            classes, palette)
        self.SEMANTIC_CLASSES, self.SEMANTIC_PALETTE = self.get_semantic_classes_and_palette(
            semantic_classes, semantic_palette)
        self.gt_seg_map_loader = MultiImgMultiAnnLoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else MultiImgMultiAnnLoadAnnotations(
            **gt_seg_map_loader_cfg)  # TODO

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if ann_dir is not None:
            assert isinstance(ann_dir, dict) and len(ann_dir) >= 3, \
                'There should be at least three anns in semantic change detection'
            # `binary_dir`, `semantic_dir_from`, `semantic_dir_to`

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if self.ann_dir is not None:
                for ann_key in self.ann_dir.keys():
                    if not osp.isabs(self.ann_dir[ann_key]):
                        self.ann_dir[ann_key] = \
                            osp.join(self.data_root, self.ann_dir[ann_key])
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.sub_dir_1, self.sub_dir_2,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def get_gt_seg_map_by_idx(self, index):
        """Get three ground truth segmentation maps for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return dict(
            gt_semantic_seg=results['gt_semantic_seg'], \
            gt_semantic_seg_from=results['gt_semantic_seg_from'], \
            gt_semantic_seg_to=results['gt_semantic_seg_to'])

    def inverse_reduce_zero_label(self, results):
        """Recover zero in semantic labels."""
        inverse_zero_results = []
        for result in results:
            binary_, from_, to_ = result
            from_, to_ = from_ + 1, to_ + 1
            from_ = from_ * binary_
            to_ = to_ * binary_
            inverse_zero_results.append([binary_, from_, to_])
        return inverse_zero_results

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): binary_cd_logit,
            semantic_cd_logit_from, semantic_cd_logit_to
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        assert len(preds[0]) == 3, '`preds` should contain `binary_cd_logit`, \
            `semantic_cd_logit_from` and `semantic_cd_logit_to`'

        if not isinstance(indices, list):
            indices = [indices]

        if not isinstance(preds, list):
            preds = [preds]

        # (B, 3, H, W)
        if self.inverse_semantic_zero_pred:
            preds = self.inverse_reduce_zero_label(preds)

        for dub_idx, sub_preds in enumerate(preds):
            if not isinstance(sub_preds, list):
                preds[dub_idx] = [sub_preds]

        pre_eval_results = []
        gt_names = ['gt_semantic_seg', \
                    'gt_semantic_seg_from', 'gt_semantic_seg_to']

        for pred, index in zip(preds, indices):
            sub_pre_eval_results = []
            seg_maps = self.get_gt_seg_map_by_idx(index)
            for sub_name, sub_pred in zip(gt_names, pred):
                if sub_name == 'gt_semantic_seg':
                    num_classes = len(self.CLASSES)
                    reduce_zero_label = self.reduce_zero_label  # False
                else:
                    num_classes = len(self.SEMANTIC_CLASSES)
                    reduce_zero_label = self.reduce_semantic_zero_label
                sub_pre_eval_results.append(
                    intersect_and_union(
                        sub_pred,
                        seg_maps[sub_name],
                        num_classes,
                        self.ignore_index,
                        # as the labels has been converted when dataset initialized
                        # in `get_palette_for_custom_classes ` this `label_map`
                        # should be `dict()`, see
                        # https://github.com/open-mmlab/mmsegmentation/issues/1415
                        # for more ditails
                        label_map=dict(),
                        reduce_zero_label=reduce_zero_label))
            pre_eval_results.append(sub_pre_eval_results)

        return pre_eval_results

    def get_binary_classes_and_palette(self, classes=None, palette=None):
        """Get binary class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """

        return classes, palette

    def get_semantic_classes_and_palette(self, classes=None, palette=None):
        """Get semantic class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.SEMANTIC_CLASSES, self.SEMANTIC_PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.SEMANTIC_CLASSES:
            if not set(class_names).issubset(self.SEMANTIC_CLASSES):
                raise ValueError('semantic_classes is not a \
                    subset of SEMANTIC_CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.SEMANTIC_CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_Sek(self, pre_eval_results):
        """calculate the Sek value.

        Args:
            pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric

        Returns:
            [torch.tensor]: The Sek value.
        """
        pre_eval_results = tuple(zip(*pre_eval_results))
        assert len(pre_eval_results) == 4

        hist_00 = sum(pre_eval_results[0])[0]

        hist_00_list = torch.zeros(len(pre_eval_results[0][0]))
        hist_00_list[0] = hist_00

        total_area_intersect = sum(pre_eval_results[0]) - hist_00_list
        total_area_pred_label = sum(pre_eval_results[2]) - hist_00_list
        total_area_label = sum(pre_eval_results[3]) - hist_00_list

        # foreground
        fg_intersect_sum = total_area_label[1:].sum(
        ) - total_area_pred_label[0]
        fg_area_union_sum = total_area_label.sum()

        po = total_area_intersect.sum() / total_area_label.sum()
        pe = (total_area_label * total_area_pred_label).sum() / \
            total_area_pred_label.sum() ** 2

        kappa0 = (po - pe) / (1 - pe)
        # the `iou_fg` is equal to the binary `changed` iou.
        iou_fg = fg_intersect_sum / fg_area_union_sum
        Sek = (kappa0 * torch.exp(iou_fg)) / torch.e

        return Sek.numpy() # consistent with other metrics.

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            raise NotImplementedError(f'Currently only `pre_eval` mode is supported')
            # if gt_seg_maps is None:
            #     gt_seg_maps = self.get_gt_seg_maps()
            # num_classes = len(self.CLASSES)
            # ret_metrics = eval_metrics(
            #     results,
            #     gt_seg_maps,
            #     num_classes,
            #     self.ignore_index,
            #     metric,
            #     label_map=dict(),
            #     reduce_zero_label=self.reduce_semantic_zero_label)
        # test a list of pre_eval_results
        else:
            # (B, 3, 4) -> (3, B, 4) for splitting the list
            results = [list(pred) for pred in list(zip(*results))]
            binary_ret_metrics = pre_eval_to_metrics(results[0], metric)
            semantic_ret_metrics = pre_eval_to_metrics(results[1] \
                                                       + results[2], metric)
            # Before upgrating to open-mmlab 2.x, calculate Sek/Kappa here temporarily.
            Sek = self.get_Sek(results[1] + results[2])

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            pass
            # class_names = tuple(range(num_classes))
        else:
            binary_class_names = self.CLASSES
            semantic_class_names = self.SEMANTIC_CLASSES

        # summary table
        binary_ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in binary_ret_metrics.items()
        })
        semantic_ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in semantic_ret_metrics.items()
        })
        # for semantic change detection
        semantic_ret_metrics_summary.update({'Sek': np.round(Sek * 100, 2)})
        semantic_ret_metrics_summary.update({'SCD_Score': \
            np.round(0.3 * binary_ret_metrics_summary['IoU'] + 0.7 * Sek * 100, 2)})

        # each class table
        binary_ret_metrics.pop('aAcc', None)
        binary_ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in binary_ret_metrics.items()
        })
        binary_ret_metrics_class.update({'Class': binary_class_names})
        binary_ret_metrics_class.move_to_end('Class', last=False)

        semantic_ret_metrics.pop('aAcc', None)
        semantic_ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in semantic_ret_metrics.items()
        })
        semantic_ret_metrics_class.update({'Class': semantic_class_names})
        semantic_ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        binary_class_table_data = PrettyTable()
        for key, val in binary_ret_metrics_class.items():
            binary_class_table_data.add_column(key, val)

        binary_summary_table_data = PrettyTable()
        for key, val in binary_ret_metrics_summary.items():
            if key == 'aAcc':
                binary_summary_table_data.add_column(key, [val])
            else:
                binary_summary_table_data.add_column('m' + key, [val])

        print_log('Binary per class results:', logger)
        print_log('\n' + binary_class_table_data.get_string(), logger=logger)
        print_log('Binary Change Detection Summary:', logger)
        print_log('\n' + binary_summary_table_data.get_string(), logger=logger)

        semantic_class_table_data = PrettyTable()
        for key, val in semantic_ret_metrics_class.items():
            semantic_class_table_data.add_column(key, val)

        semantic_summary_table_data = PrettyTable()
        for key, val in semantic_ret_metrics_summary.items():
            if key == 'aAcc':
                semantic_summary_table_data.add_column(key, [val])
            else:
                semantic_summary_table_data.add_column('m' + key, [val])

        print_log('Semantic per class results:', logger)
        print_log('\n' + semantic_class_table_data.get_string(), logger=logger)
        print_log('Semantic Change Detection Summary:', logger)
        print_log(
            '\n' + semantic_summary_table_data.get_string(), logger=logger)

        # each metric dict
        binary_eval_results = {}
        for key, value in binary_ret_metrics_summary.items():
            if key == 'aAcc':
                binary_eval_results[key] = value / 100.0
            else:
                binary_eval_results['m' + key] = value / 100.0

        binary_ret_metrics_class.pop('Class', None)
        for key, value in binary_ret_metrics_class.items():
            binary_eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(binary_class_names)
            })

        eval_results.update(add_prefix(binary_eval_results, 'binary'))

        semantic_eval_results = {}
        for key, value in semantic_ret_metrics_summary.items():
            if key == 'aAcc':
                semantic_eval_results[key] = value / 100.0
            else:
                semantic_eval_results['m' + key] = value / 100.0

        semantic_ret_metrics_class.pop('Class', None)
        for key, value in semantic_ret_metrics_class.items():
            semantic_eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(semantic_class_names)
            })
        eval_results.update(add_prefix(semantic_eval_results, 'semantic'))

        return eval_results
