# CGNet

[Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery](https://ieeexplore.ieee.org/document/10234560)

## Introduction

[Official Repo](https://github.com/ChengxiHAN/CGNet-CD)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/cgnet.py)

## Abstract
The rapid advancement of automated artificial intelligence algorithms and remote sensing instruments has benefited change detection (CD) tasks. However, there is still a lot of space to study for precise detection, especially the edge integrity and internal holes phenomenon of change features. In order to solve these problems, we design the change guiding network (CGNet) to tackle the insufficient expression problem of change features in the conventional U-Net structure adopted in previous methods, which causes inaccurate edge detection and internal holes. Change maps from deep features with rich semantic information are generated and used as prior information to guide multiscale feature fusion, which can improve the expression ability of change features. Meanwhile, we propose a self-attention module named change guide module, which can effectively capture the long-distance dependency among pixels and effectively overcomes the problem of the insufficient receptive field of traditional convolutional neural networks. On four major CD datasets, we verify the usefulness and efficiency of the CGNet, and a large number of experiments and ablation studies demonstrate the effectiveness of CGNet.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/likyoo/open-cd/assets/44317497/d4f0e42e-8446-448f-8afa-d1ab4339283d" width="75%"/>
</div>



```bibtex
@ARTICLE{10234560,
  author={Han, Chengxi and Wu, Chen and Guo, Haonan and Hu, Meiqi and Li, Jiepan and Chen, Hongruixuan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery}, 
  year={2023},
  volume={16},
  number={},
  pages={8395-8407},
  keywords={Feature extraction;Transformers;Convolutional neural networks;Remote sensing;Deep learning;Decoding;Computational modeling;Artificial intelligence;Change detection (CD);change guide module (CGM);change guiding map;deep learning;high-resolution remote sensing (RS) image},
  doi={10.1109/JSTARS.2023.3310208}}
```

## Results and models

### LEVIR-CD-256

| Method | Image Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :--------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| CGNet  |  256x256   |  100e   |    -     |   93.18   | 90.99  |  92.07   | 85.31 | [config](https://github.com/likyoo/open-cd/blob/main/configs/cgnet/cgnet_256x256_100e_levircd-256.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
