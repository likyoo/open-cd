# FCSN

[Fully Convolutional Siamese Networks for Change Detection](https://arxiv.org/abs/1810.08462)

## Introduction

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/siamunet_diff.py)

## Abstract
This paper presents three fully convolutional neural network architectures which perform change detection using a pair of coregistered images. Most notably, we propose two Siamese extensions of fully convolutional networks which use heuristics about the current problem to achieve the best results in our tests on two open change detection datasets, using both RGB and multispectral images. We show that our system is able to learn from scratch using annotated change detection images. Our architectures achieve better performance than previously proposed methods, while being at least 500 times faster than related systems. This work is a step towards efficient processing of data from large scale Earth observation systems such as Copernicus or Landsat.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/201501311-a5782a63-cf41-4ac3-bcc3-bcdc2612fc69.png" width="90%"/>
</div>

```bibtex
@inproceedings{daudt2018fully,
  title={Fully convolutional siamese networks for change detection},
  author={Daudt, Rodrigo Caye and Le Saux, Bertr and Boulch, Alexandre},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={4063--4067},
  year={2018},
  organization={IEEE}
}
```

## Results and models

### LEVIR-CD

|    Method    | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
|    FC-EF     |  256x256  |  40000  |    -     |   87.47   | 84.28  |  85.84   | 75.20 | [config](https://github.com/likyoo/open-cd/blob/main/configs/fcsn/fc_ef_256x256_40k_levircd.py) |          |
| FC-Siam-Diff |  256x256  |  40000  |    -     |   91.14   | 83.78  |  87.31   | 77.47 | [config](https://github.com/likyoo/open-cd/blob/main/configs/fcsn/fc_siam_diff_256x256_40k_levircd.py) |          |
| FC-Siam-Conc |  256x256  |  40000  |    -     |   88.08   | 88.95  |  88.51   | 79.39 | [config](https://github.com/likyoo/open-cd/blob/main/configs/fcsn/fc_siam_conc_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.