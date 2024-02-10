# LightCDNet

[LightCDNet: Lightweight Change Detection Network Based on VHR Images](https://ieeexplore.ieee.org/document/10214556)

## Introduction

[Official Repo](https://github.com/NightSongs/LightCDNet)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/lightcdnet.py)

## Abstract
Lightweight change detection models are essential for industrial applications and edge devices. Reducing the model size while maintaining high accuracy is a key challenge in developing lightweight change detection models. However, many existing methods oversimplify the model architecture, leading to a loss of information and reduced performance. Therefore, developing a lightweight model that can effectively preserve the input information is a challenging problem. To address this challenge, we propose LightCDNet, a novel lightweight change detection model that effectively preserves the input information. LightCDNet consists of an early fusion backbone network and a pyramid decoder for end-to-end change detection. The core component of LightCDNet is the Deep Supervised Fusion Module (DSFM), which guides the early fusion of primary features to improve performance. We evaluated LightCDNet on the LEVIR-CD dataset and found that it achieved comparable or better performance than state-of-the-art models while being 10–117 times smaller in size.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/likyoo/open-cd/assets/44317497/cec088ca-cb45-4d32-8ebb-c0fd3b8d1a4c" width="90%"/>
</div>


```bibtex
@ARTICLE{10214556,
  author={Xing, Yuanjun and Jiang, Jiawei and Xiang, Jun and Yan, Enping and Song, Yabin and Mo, Dengkui},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={LightCDNet: Lightweight Change Detection Network Based on VHR Images}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3304309}}
```

## Results and models

### LEVIR-CD

|      Method      | Crop Size | Lr schd | \#Param (M) | MACs (G) | Precision | Recall | F1-Score |  IoU  | config                                                       |
| :--------------: | :-------: | :-----: | :---------: | :------: | :-------: | :----: | :------: | :---: | ------------------------------------------------------------ |
| LightCDNet-small |  256x256  |  40000  |    0.35     |   1.65   |   91.36   | 89.81  |  90.57   | 82.77 | [config](https://github.com/likyoo/open-cd/blob/main/configs/lightcdnet/lightcdnet_s_256x256_40k_levircd.py) |
| LightCDNet-base  |  256x256  |  40000  |    1.32     |   3.22   |   92.12   | 90.43  |  91.27   | 83.94 | [config](https://github.com/likyoo/open-cd/blob/main/configs/lightcdnet/lightcdnet_b_256x256_40k_levircd.py) |
| LightCDNet-large |  256x256  |  40000  |    2.82     |   5.94   |   92.43   | 90.45  |  91.43   | 84.21 | [config](https://github.com/likyoo/open-cd/blob/main/configs/lightcdnet/lightcdnet_l_256x256_40k_levircd.py) |


- All metrics are based on the category "change".
- All scores are computed on the test set.
