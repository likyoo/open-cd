import csv
import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def get_metrics(total_mat):
    """
    total_mat: a total confusion matrix
    """
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    iou = np.diag(total_mat) / (total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    precision = np.diag(total_mat) / total_mat.sum(axis=0)
    recall = np.diag(total_mat) / total_mat.sum(axis=1)
    F1 = 2*precision*recall / (precision + recall)

    return all_acc, iou[1], precision[1], recall[1], F1[1]


def read_img(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    img[img < 128] = 0
    img[img >= 128] = 1
    return img


if __name__ == '__main__':
    pred_root = '/home/dml307/exp/likyoo/result_data/result_opencd_baseline/infer_res'
    label_root = '/home/dml307/data/cd_dataset/LEVIR-CD/test/label'
    
    # total_mat = np.zeros((2, 2), dtype=float)
    # num_classes = 2
    # for img_name in tqdm(os.listdir(pred_root)):
        
    #     pred = read_img(osp.join(pred_root, img_name))
    #     gt = read_img(osp.join(label_root, img_name))
        
    #     mask = (gt >= 0) & (gt < num_classes)
    #     mat = np.bincount(2 * gt[mask].astype(int) + pred[mask],
    #                        minlength=num_classes**2).reshape(num_classes, num_classes)

    #     total_mat += mat
        
    # all_acc, iou, precision, recall, F1 = get_metrics(total_mat)
    
    # print(f"all_acc: {all_acc},\niou: {iou},\nprecision: {precision},\nrecall: {recall}, \nF1: {F1}")
    
    FILE_METRIC_LIST = []
    num_classes = 2
    for img_name in tqdm(os.listdir(pred_root)):
        
        pred = read_img(osp.join(pred_root, img_name))
        gt = read_img(osp.join(label_root, img_name))
        
        mask = (gt >= 0) & (gt < num_classes)
        mat = np.bincount(num_classes * gt[mask].astype(int) + pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        
        FP, FN = mat[0, 1], mat[1, 0]
        all_acc, iou, precision, recall, F1 = get_metrics(mat)
        FILE_METRIC_LIST.append([img_name, F1, precision, recall, iou, FP, FN])
        
    with open('score.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'F1', 'precision', 'recall', 'iou', 'FT', 'FN'])
        writer.writerows(FILE_METRIC_LIST)
        