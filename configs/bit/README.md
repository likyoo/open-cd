# BiT

[Remote Sensing Image Change Detection with Transformers](https://arxiv.org/abs/2103.00208)

## Introduction

[Official Repo](https://github.com/justchenhao/BIT_CD)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/bit_head.py)

## Abstract
Modern change detection (CD) has achieved remarkable success by the powerful discriminative ability of deep convolutions. However, high-resolution remote sensing CD remains challenging due to the complexity of objects in the scene. Objects with the same semantic concept may show distinct spectral characteristics at different times and spatial locations. Most recent CD pipelines using pure convolutions are still struggling to relate long-range concepts in space-time. Nonlocal self-attention approaches show promising performance via modeling dense relationships among pixels, yet are computationally inefficient. Here, we propose a bitemporal image transformer (BIT) to efficiently and effectively model contexts within the spatial-temporal domain. Our intuition is that the high-level concepts of the change of interest can be represented by a few visual words, that is, semantic tokens. To achieve this, we express the bitemporal image into a few tokens and use a transformer encoder to model contexts in the compact token-based space-time. The learned context-rich tokens are then fed back to the pixel-space for refining the original features via a transformer decoder. We incorporate BIT in a deep feature differencing-based CD framework. Extensive experiments on three CD datasets demonstrate the effectiveness and efficiency of the proposed method. Notably, our BIT-based model significantly outperforms the purely convolutional baseline using only three times lower computational costs and model parameters. Based on a naive backbone (ResNet18) without sophisticated structures (e.g., feature pyramid network (FPN) and UNet), our model surpasses several state-of-the-art CD methods, including better than four recent attention-based methods in terms of efficiency and accuracy. Our code is available at https://github.com/justchenhao/BIT_CD.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/201470502-b50219fa-0b54-479e-9be1-836b5a5026c8.png" width="90%"/>
</div>

```bibtex
@Article{chen2021a,
    title={Remote Sensing Image Change Detection with Transformers},
    author={Hao Chen, Zipeng Qi and Zhenwei Shi},
    year={2021},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095166}
}
```

## Results and models

### LEVIR-CD

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
|  BiT   |   r18    |  256x256  |  40000  |    -     |   91.97   | 88.62  |  90.26   | 82.25 | [config](https://github.com/likyoo/open-cd/blob/main/configs/bit/bit_r18_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
- The stage4 of resnet-18 is removed.

