# SNUNet

[SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images](https://ieeexplore.ieee.org/document/9355573)

## Introduction

[Official Repo](https://github.com/likyoo/Siam-NestedUNet)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/snunet.py)

## Abstract
Change detection is an important task in remote sensing (RS) image analysis. It is widely used in natural disaster monitoring and assessment, land resource planning, and other fields. As a pixel-to-pixel prediction task, change detection is sensitive about the utilization of the original position information. Recent change detection methods always focus on the extraction of deep change semantic feature, but ignore the importance of shallow-layer information containing high-resolution and finegrained features, this often leads to the uncertainty of the pixels at the edge of the changed target and the determination miss of small targets. In this letter, we propose a densely connected siamese network for change detection, namely SNUNet-CD (the combination of Siamese network and NestedUNet). SNUNet-CD alleviates the loss of localization information in the deep layers of neural network through compact information transmission between encoder and decoder, and between decoder and decoder. In addition, Ensemble Channel Attention Module (ECAM) is proposed for deep supervision. Through ECAM, the most representative features of different semantic levels can be refined and used for the final classification. Experimental results show that our method improves greatly on many evaluation criteria and has a better tradeoff between accuracy and calculation amount than other state-of-the-art (SOTA) change detection methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/201501845-da98c364-e0fe-4c75-be8b-f9d207e993f5.png" width="90%"/>
</div>

```bibtex
@ARTICLE{9355573,
  author={S. {Fang} and K. {Li} and J. {Shao} and Z. {Li}},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images}, 
  year={2021},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2021.3056416}}
```

## Results and models

### LEVIR-CD

| Method | base_channel | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :----------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| SNUNet |      16      |  256x256  |  40000  |    -     |   92.70   | 90.04  |  91.35   | 84.08 | [config](https://github.com/likyoo/open-cd/blob/main/configs/snunet/snunet_c16_256x256_40k_levircd.py) |          |


### SVCD

| Method | base_channel | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :----------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| SNUNet |      16      |  256x256  |  120000  |    -     |   94.69   | 91.90  |  93.27   | 87.40 | [config](https://github.com/likyoo/open-cd/blob/main/configs/snunet/snunet_c16_256x256_120k_svcd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
- 120000 iters ~ 100 epochs in SVCD Dataset
