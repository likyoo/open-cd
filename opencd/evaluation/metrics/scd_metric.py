# Copyright (c) Open-CD. All rights reserved.
import copy
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.evaluation import IoUMetric
from opencd.registry import METRICS


@METRICS.register_module()
class SCDMetric(IoUMetric):
    """Change Detection evaluation metric.

    Args:
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to 'binary'.
        semantic_prefix (str, optional): The prefix that will be added in the 
            metric names to disambiguate homonymous metrics of different 
            evaluators. Defaults to 'semantic'.
        cal_sek bool: Whether to calculate the separated kappa (SeK) 
            coefficient. Defaults: False.
    """

    def __init__(self,
                 prefix: Optional[str] = 'binary',
                 semantic_prefix: Optional[str] = 'semantic',
                 cal_sek: bool = False,
                 **kwargs) -> None:
        super().__init__(prefix=prefix, **kwargs)

        self.semantic_results: List[Any] = []
        self.semantic_prefix = semantic_prefix
        self.cal_sek = cal_sek

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        num_semantic_classes = len(self.dataset_meta['semantic_classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            pred_label_from = data_sample['pred_sem_seg_from']['data'].squeeze()
            label_from = data_sample['gt_sem_seg_from']['data'].squeeze().to(pred_label_from)
            pred_label_to = data_sample['pred_sem_seg_to']['data'].squeeze()
            label_to = data_sample['gt_sem_seg_to']['data'].squeeze().to(pred_label_to)

            self.results.append(
                self.intersect_and_union(pred_label, label, num_classes,
                                         self.ignore_index))
            # for semantic pred
            self.semantic_results.append(
                self.intersect_and_union(pred_label_from, label_from, num_semantic_classes,
                                         self.ignore_index))
            self.semantic_results.append(
                self.intersect_and_union(pred_label_to, label_to, num_semantic_classes,
                                         self.ignore_index))

    def get_sek(self, results: list) -> np.array:
        """calculate the Sek value.

        Args:
            pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric

        Returns:
            [torch.tensor]: The Sek value.
        """
        assert len(results) == 4

        hist_00 = sum(results[0])[0]

        hist_00_list = torch.zeros(len(results[0][0]))
        hist_00_list[0] = hist_00

        total_area_intersect = sum(results[0]) - hist_00_list
        total_area_pred_label = sum(results[2]) - hist_00_list
        total_area_label = sum(results[3]) - hist_00_list

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
        sek = (kappa0 * torch.exp(iou_fg)) / torch.e

        return sek.numpy() # consistent with other metrics.
    
    def compute_metrics(self, binary_results: list, semantic_results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            binary_results (list): The processed results of each batch.
            semantic_results (list): The semantic results of each batch

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        binary_results = tuple(zip(*binary_results))
        semantic_results = tuple(zip(*semantic_results))
        assert len(binary_results) == 4 and len(semantic_results) == 4

        # for binary results
        binary_total_area_intersect = sum(binary_results[0])
        binary_total_area_union = sum(binary_results[1])
        binary_total_area_pred_label = sum(binary_results[2])
        binary_total_area_label = sum(binary_results[3])
        binary_ret_metrics = self.total_area_to_metrics(
            binary_total_area_intersect, binary_total_area_union, binary_total_area_pred_label,
            binary_total_area_label, self.metrics, self.nan_to_num, self.beta)

        binary_class_names = self.dataset_meta['classes']

        # summary table
        binary_ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in binary_ret_metrics.items()
        })
        binary_metrics = dict()
        for key, val in binary_ret_metrics_summary.items():
            if key == 'aAcc':
                binary_metrics[key] = val
            else:
                binary_metrics['m' + key] = val

        # each class table
        binary_ret_metrics.pop('aAcc', None)
        binary_ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in binary_ret_metrics.items()
        })
        binary_ret_metrics_class.update({'Class': binary_class_names})
        binary_ret_metrics_class.move_to_end('Class', last=False)
        binary_class_table_data = PrettyTable()
        for key, val in binary_ret_metrics_class.items():
            binary_class_table_data.add_column(key, val)

        print_log('per binary class results:', logger)
        print_log('\n' + binary_class_table_data.get_string(), logger=logger)

        # for semantic results
        semantic_total_area_intersect = sum(semantic_results[0])
        semantic_total_area_union = sum(semantic_results[1])
        semantic_total_area_pred_label = sum(semantic_results[2])
        semantic_total_area_label = sum(semantic_results[3])
        semantic_ret_metrics = self.total_area_to_metrics(
            semantic_total_area_intersect, semantic_total_area_union, semantic_total_area_pred_label,
            semantic_total_area_label, self.metrics, self.nan_to_num, self.beta)

        semantic_class_names = self.dataset_meta['semantic_classes']

        # summary table
        semantic_ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in semantic_ret_metrics.items()
        })
        # for semantic change detection
        if self.cal_sek:
            sek = self.get_sek(semantic_results)
            semantic_ret_metrics_summary.update({'Sek': np.round(sek * 100, 2)})
            semantic_ret_metrics_summary.update({'SCD_Score': \
                np.round(0.3 * binary_ret_metrics_summary['IoU'] + 0.7 * sek * 100, 2)})
        
        semantic_metrics = dict()
        for key, val in semantic_ret_metrics_summary.items():
            if key in ['aAcc', 'Sek', 'SCD_Score']:
                semantic_metrics[key] = val
            else:
                semantic_metrics['m' + key] = val

        # each class table
        semantic_ret_metrics.pop('aAcc', None)
        semantic_ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in semantic_ret_metrics.items()
        })
        semantic_ret_metrics_class.update({'Class': semantic_class_names})
        semantic_ret_metrics_class.move_to_end('Class', last=False)
        semantic_class_table_data = PrettyTable()
        for key, val in semantic_ret_metrics_class.items():
            semantic_class_table_data.add_column(key, val)

        print_log('per semantic class results:', logger)
        print_log('\n' + semantic_class_table_data.get_string(), logger=logger)

        return binary_metrics, semantic_metrics

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)
        if len(self.semantic_results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.semantic_results`. '
                'Please ensure that the processed results are properly added '
                'into `self.semantic_results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        binary_results = collect_results(self.results, size, self.collect_device)
        semantic_results = collect_results(self.semantic_results, \
                                           size * 2, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            binary_results = _to_cpu(binary_results)
            semantic_results = _to_cpu(semantic_results)
            _binary_metrics, _semantic_metrics = \
                self.compute_metrics(binary_results, semantic_results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _binary_metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _binary_metrics.items()
                }
                _semantic_metrics = {
                    '/'.join((self.semantic_prefix, k)): v
                    for k, v in _semantic_metrics.items()
                }
                _metrics = {**_binary_metrics, **_semantic_metrics}
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        self.semantic_results.clear()
        return metrics[0]
