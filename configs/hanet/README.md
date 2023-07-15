# HANet

[HANet: A Hierarchical Attention Network for Change Detection With Bitemporal Very-High-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/10093022)

## Introduction

[Official Repo](https://github.com/ChengxiHAN/HANet-CD)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/hanet.py)

## Abstract
Benefiting from the developments in deep learning technology, deep-learning-based algorithms employing automatic feature extraction have achieved remarkable performance on the change detection (CD) task. However, the performance of existing deep-learning-based CD methods is hindered by the imbalance between changed and unchanged pixels. To tackle this problem, a progressive foreground-balanced sampling strategy on the basis of not adding change information is proposed in this article to help the model accurately learn the features of the changed pixels during the early training process and thereby improve detection performance. Furthermore, we design a discriminative Siamese network, hierarchical attention network (HANet), which can integrate multiscale features and refine detailed features. The main part of HANet is the HAN module, which is a lightweight and effective self-attention mechanism. Extensive experiments and ablation studies on two CD datasets with extremely unbalanced labels validate the effectiveness and efficiency of the proposed method.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/likyoo/open-cd/assets/44317497/3b2d139e-35db-4691-87da-a1bb87819454" width="90%"/>
</div>

```bibtex
@ARTICLE{10093022,
  author={Han, Chengxi and Wu, Chen and Guo, Haonan and Hu, Meiqi and Chen, Hongruixuan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={HANet: A Hierarchical Attention Network for Change Detection With Bitemporal Very-High-Resolution Remote Sensing Images}, 
  year={2023},
  volume={16},
  number={},
  pages={3867-3878},
  doi={10.1109/JSTARS.2023.3264802}}

```

## Results and models

### LEVIR-CD

| Method | PFBS | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :--: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| HANet  | w/o  |  256x256  |  40000  |    -     |   91.73   | 90.06  |  90.89   | 83.29 | [config](https://github.com/likyoo/open-cd/blob/main/configs/hanet/hanet_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
- `PFBS` indicates Progressive Foreground-Balanced Sampling.
