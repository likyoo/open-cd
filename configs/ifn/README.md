# IFN

[A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images](https://www.sciencedirect.com/science/article/pii/S0924271620301532)

## Introduction

[Official Repo](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/ifn.py)

## Abstract
Change detection in high resolution remote sensing images is crucial to the understanding of land surface changes. As traditional change detection methods are not suitable for the task considering the challenges brought by the fine image details and complex texture features conveyed in high resolution images, a number of deep learning-based change detection methods have been proposed to improve the change detection performance. Although the state-of-the-art deep feature based methods outperform all the other deep learning-based change detection methods, networks in the existing deep feature based methods are mostly modified from architectures that are originally proposed for single-image semantic segmentation. Transferring these networks for change detection task still poses some key issues. In this paper, we propose a deeply supervised image fusion network (IFN) for change detection in high resolution bi-temporal remote sensing images. Specifically, highly representative deep features of bi-temporal images are firstly extracted through a fully convolutional two-stream architecture. Then, the extracted deep features are fed into a deeply supervised difference discrimination network (DDN) for change detection. To improve boundary completeness and internal compactness of objects in the output change maps, multi-level deep features of raw images are fused with image difference features by means of attention modules for change map reconstruction. DDN is further enhanced by directly introducing change map losses to intermediate layers in the network, and the whole network is trained in an end-to-end manner. IFN is applied to a publicly available dataset, as well as a challenging dataset consisting of multi-source bi-temporal images from Google Earth covering different cities in China. Both visual interpretation and quantitative assessment confirm that IFN outperforms four benchmark methods derived from the literature, by returning changed areas with complete boundaries and high internal compactness compared to the state-of-the-art methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/215308284-cafb5fa7-0c1f-404e-804b-71dff28e1b63.png" width="90%"/>
</div>

```bibtex
@article{zhang2020deeply,
  title={A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images},
  author={Zhang, Chenxiao and Yue, Peng and Tapete, Deodato and Jiang, Liangcun and Shangguan, Boyi and Huang, Li and Liu, Guangchao},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={166},
  pages={183--200},
  year={2020},
  publisher={Elsevier}
}
```

## Results and models

### LEVIR-CD

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
|  IFN   |  vgg16   |  256x256  |  40000  |    -     |   91.17   | 90.51  |  90.83   | 83.21 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ifn/ifn_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
