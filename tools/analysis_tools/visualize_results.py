import os
import os.path as osp
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class CDVisualization(object):
    def __init__(self, policy=['compare_pixel', 'pixel']):
        """Change Detection Visualization

        Args:
            policy (list, optional): _description_. Defaults to ['compare_pixel', 'pixel'].
        """
        super().__init__()
        assert isinstance(policy, list)
        self.policy = policy
        self.num_classes = 2
        self.COLOR_MAP = {'0': (0, 0, 0), # black is TN
                          '1': (255, 255, 0), # yellow is FP 误检
                          '2': (255, 0, 0), # red is FN 漏检
                          '3':(0, 255, 0)} # green is TP
        
    
    def read_and_check_label(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if np.max(img) >= self.num_classes:
            warnings.warn('Please make sure the range of the pixel value' \
                  'in your pred or gt.')
            img[img < 128] = 0
            img[img >= 128] = 1
        return img
    
    def read_img(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def save_img(self, file_name, vis_res, imgs=None):
        dir_name = os.path.dirname(file_name)
        base_name, suffix = os.path.basename(file_name).split('.')
        os.makedirs(dir_name, exist_ok=True)
        
        # consistent with the original image
        vis_res = cv2.cvtColor(vis_res, cv2.COLOR_RGB2BGR)
        if imgs is not None:
            assert isinstance(imgs, list), '`imgs` must be a list.'
            if vis_res.shape != imgs[0].shape:
                vis_res = cv2.cvtColor(vis_res, cv2.COLOR_GRAY2RGB)
            for idx, img in enumerate(imgs):
                assert img.shape == vis_res.shape, '`img` and `vis_res` must be '\
                    'of the same shape.'
                cv2.imwrite(osp.join(dir_name, base_name+'_'+str(idx)+'.'+suffix), 
                            cv2.addWeighted(img, 1, vis_res, 0.25, 0.0))
        else:
            cv2.imwrite(file_name, vis_res)

    def trainIdToColor(self, trainId):
        """convert label id to color

        Args:
            trainId (int): _description_

        Returns:
            color (tuple)
        """
        color = self.COLOR_MAP[str(trainId)]
        return color

    def gray2color(self, grayImage: np.ndarray, num_class: list):
        """convert label to color image

        Args:
            grayImage (np.ndarray): _description_
            num_class (list): _description_

        Returns:
            _type_: _description_
        """
        rgbImage=np.zeros((grayImage.shape[0],grayImage.shape[1],3),dtype='uint8')
        for cls in num_class:
            row, col=np.where(grayImage==cls)
            if (len(row)==0):
                continue
            color=self.trainIdToColor(cls)
            rgbImage[row, col]=color
        return rgbImage
    
    def res_pixel_visual(self, label):
        assert np.max(label) < self.num_classes, 'There exists the value of ' \
        '`label` that is greater than `num_classes`'
        label_rgb = self.gray2color(label, num_class=list(range(self.num_classes)))
        return label_rgb
    
    def res_compare_pixel_visual(self, pred, gt):
        """visualize according to confusion matrix.

        Args:
            pred (_type_): _description_
            gt (_type_): _description_
        """
        assert np.max(pred) < self.num_classes, 'There exists the value of ' \
        '`pred` that is greater than `num_classes`'
        assert np.max(gt) < self.num_classes, 'There exists the value of ' \
        '`gt` that is greater than `num_classes`'
        
        visual_ = self.num_classes * gt.astype(int) + pred.astype(int)
        visual_rgb = self.gray2color(visual_, num_class=list(range(self.num_classes**2)))
        return visual_rgb

    def res_compare_boundary_visual(self):
        pass

    def __call__(self, pred_path, gt_path, dst_path, imgs=None):
        # dst_prefix, dst_suffix = osp.abspath(dst_path).split('.')
        # dst_path = dst_prefix + '_{}.' + dst_suffix
        file_name = osp.basename(dst_path)
        dst_path = osp.dirname(dst_path)
        
        pred = self.read_and_check_label(pred_path)
        gt = self.read_and_check_label(gt_path)
        if imgs is not None:
            assert isinstance(imgs, list), '`imgs` must be a list.'
            imgs = [self.read_img(p) for p in imgs]
        
        for pol in self.policy:
            dst_path_pol = osp.join(dst_path, pol)
            os.makedirs(dst_path_pol, exist_ok=True)
            dst_file = osp.join(dst_path_pol, file_name)
            if pol == 'compare_pixel':
                visual_map = self.res_compare_pixel_visual(pred, gt)
                self.save_img(dst_file, visual_map, None)
            elif pol == 'pixel':
                visual_map = self.res_pixel_visual(pred)
                self.save_img(dst_file, visual_map, imgs)
            else:
                raise ValueError(f'Invalid policy {pol}')
            

if __name__ == '__main__':
    gt_dir = '/home/dml307/data/cd_dataset/S2Looking/test/label'
    pred_dir = '/home/dml307/exp/likyoo/tmp_data/infer_res'
    dst_dir = './tmp_visual'
    img_dir = ['/home/dml307/data/cd_dataset/S2Looking/test/Image1', 
               '/home/dml307/data/cd_dataset/S2Looking/test/Image2']
    file_name = '127.png'
    
    CDVisual = CDVisualization(policy=['compare_pixel'])
    CDVisual(osp.join(pred_dir, file_name), 
             osp.join(gt_dir, file_name), 
             osp.join(dst_dir, file_name),
             [osp.join(p, file_name) for p in img_dir])
