# STANet

[A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)

## Introduction

[Official Repo](https://github.com/justchenhao/STANet)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/sta_head.py)

## Abstract
Very-high-resolution (VHR) bi-temporal images change detection (CD) is a basic remote sensing images (RSIs) processing task. Recently, deep convolutional neural networks (DCNNs) have shown great feature representation abilities in computer vision tasks and have achieved remarkable breakthroughs in automatic CD. However, a great majority of the existing fusion-based CD methods pay no attention to the definition of CD, so they can only detect one-way changes. Therefore, we propose a new temporal reliable change detection (TRCD) algorithm to solve this drawback of fusion-based methods. Specifically, a potential and effective algorithm is proposed for learning temporal-reliable features for CD, which is achieved by designing a novel objective function. Unlike the traditional CD objective function, we impose a regular term in the objective function, which aims to enforce the extracted features before and after exchanging sequences of bi-temporal images that are similar to each other. In addition, our backbone architecture is designed based on a high-resolution network. The captured features are semantically richer and more spatially precise, which can improve the performance for small region changes. Comprehensive experimental results on two public datasets demonstrate that the proposed method is more advanced than other state-of-the-art (SOTA) methods, and our proposed objective function shows great potential.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/210191098-25c7cd63-b43c-4d4f-a549-9d9946643caa.png" width="90%"/>
</div>

```bibtex
@Article{rs12101662,
    AUTHOR = {Chen, Hao and Shi, Zhenwei},
    TITLE = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
    JOURNAL = {Remote Sensing},
    VOLUME = {12},
    YEAR = {2020},
    NUMBER = {10},
    ARTICLE-NUMBER = {1662},
    URL = {https://www.mdpi.com/2072-4292/12/10/1662},
    ISSN = {2072-4292},
    DOI = {10.3390/rs12101662}
}
```

## Results and models

### LEVIR-CD

|   Method    | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :---------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| STANet-BASE |  256x256  |  40000  |    -     |   69.20   | 96.41  |  80.57   | 67.46 | [config](https://github.com/likyoo/open-cd/blob/main/configs/stanet/stanet_base_256x256_40k_levircd.py) |          |
| STANet-BAM  |  256x256  |  40000  |    -     |   70.01   | 96.60  |  81.18   | 68.32 | [config](https://github.com/likyoo/open-cd/blob/main/configs/stanet/stanet_bam_256x256_40k_levircd.py) |          |
| STANet-PAM  |  256x256  |  40000  |    -     |   72.79   | 96.20  |  82.88   | 70.76 | [config](https://github.com/likyoo/open-cd/blob/main/configs/stanet/stanet_pam_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
