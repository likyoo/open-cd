# BAN

[A New Learning Paradigm for Foundation Model-based Remote Sensing Change Detection](https://arxiv.org/abs/2312.01163)

## Introduction

[Official Repo](https://github.com/likyoo/BAN)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/ban_head.py)

## Abstract

Change detection (CD) is a critical task to observe and analyze dynamic processes of land cover. Although numerous deep learning-based CD models have performed excellently, their further performance improvements are constrained by the limited knowledge extracted from the given labelled data. On the other hand, the foundation models that emerged recently contain a huge amount of knowledge by scaling up across data modalities and proxy tasks. In this paper, we propose a Bi-Temporal Adapter Network (BAN), which is a universal foundation model-based CD adaptation framework aiming to extract the knowledge of foundation models for CD. The proposed BAN contains three parts, i.e. frozen foundation model (e.g., CLIP), bitemporal adapter branch (Bi-TAB), and bridging modules between them. Specifically, the Bi-TAB can be either an existing arbitrary CD model or some hand-crafted stacked blocks. The bridging modules are designed to align the general features with the task/domain-specific features and inject the selected general knowledge into the Bi-TAB. To our knowledge, this is the first universal framework to adapt the foundation model to the CD task. Extensive experiments show the effectiveness of our BAN in improving the performance of existing CD methods (e.g., up to 4.08\% IoU improvement) with only a few additional learnable parameters. More importantly, these successful practices show us the potential of foundation models for remote sensing CD. The code is available at https://github.com/likyoo/BAN and will be supported in [Open-CD](https://github.com/likyoo/open-cd).

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/likyoo/BAN/blob/main/resources/BAN.png" width="100%"/>
</div>




```bibtex
@ARTICLE{10438490,
  author={Li, Kaiyu and Cao, Xiangyong and Meng, Deyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A New Learning Paradigm for Foundation Model-based Remote Sensing Change Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Adaptation models;Task analysis;Data models;Computational modeling;Feature extraction;Transformers;Tuning;Change detection;foundation model;visual tuning;remote sensing image processing;deep learning},
  doi={10.1109/TGRS.2024.3365825}}

```

## Results and models

### LEVIR-CD

| Method |       Pretrain       |     Bi-TAB      | Crop Size | Lr schd | Precision | Recall | F1-Score |  IoU  |                            config                            |
| :----: | :------------------: | :-------------: | :-------: | :-----: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: |
|  BAN   |    ViT-L/14, CLIP    |       BiT       |  512x512  |  40000  |   92.83   | 90.89  |  91.85   | 84.93 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-clip_bit_512x512_40k_levircd.py) |
|  BAN   |    ViT-B/16, CLIP    | ChangeFormer-b0 |  512x512  |  40000  |   93.25   | 90.21  |  91.71   | 84.68 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py) |
|  BAN   |    ViT-L/14, CLIP    | ChangeFormer-b0 |  512x512  |  40000  |   93.47   | 90.30  |  91.86   | 84.94 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.py) |
|  BAN   |    ViT-L/14, CLIP    | ChangeFormer-b1 |  512x512  |  40000  |   93.48   | 90.76  |  92.10   | 85.36 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-clip_mit-b1_512x512_40k_levircd.py) |
|  BAN   |    ViT-L/14, CLIP    | ChangeFormer-b2 |  512x512  |  40000  |   93.61   | 91.02  |  92.30   | 85.69 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-clip_mit-b2_512x512_40k_levircd.py) |
|  BAN   | ViT-B/32, RemoteCLIP | ChangeFormer-b0 |  512x512  |  40000  |   93.28   | 90.26  |  91.75   | 84.75 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-b32-remoteclip_mit-b0_512x512_40k_levircd.py) |
|  BAN   | ViT-L/14, RemoteCLIP | ChangeFormer-b0 |  512x512  |  40000  |   93.44   | 90.46  |  91.92   | 85.05 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-remoteclip_mit-b0_512x512_40k_levircd.py) |
|  BAN   | ViT-B/32, GeoRSCLIP  | ChangeFormer-b0 |  512x512  |  40000  |   93.35   | 90.24  |  91.77   | 84.79 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-b32-georsclip_mit-b0_512x512_40k_levircd.py) |
|  BAN   | ViT-L/14, GeoRSCLIP  | ChangeFormer-b0 |  512x512  |  40000  |   93.50   | 90.48  |  91.96   | 85.13 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-georsclip_mit-b0_512x512_40k_levircd.py) |
|  BAN   |   ViT-B/16, IN-21K   | ChangeFormer-b0 |  512x512  |  40000  |   93.59   | 89.80  |  91.66   | 84.60 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-b16-in21k_mit-b2_512x512_40k_levircd.py) |
|  BAN   |   ViT-L/16, IN-21K   | ChangeFormer-b0 |  512x512  |  40000  |   93.27   | 90.11  |  91.67   | 84.61 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l16-in21k_mit-b0_512x512_40k_levircd.py) |

### S2Looking

| Method |    Pretrain    |     Bi-TAB      | Crop Size | Lr schd | Precision | Recall | F1-Score |  IoU  |                            config                            |
| :----: | :------------: | :-------------: | :-------: | :-----: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: |
|  BAN   | ViT-L/14, CLIP |       BiT       |  512x512  |  80000  |   75.06   | 58.00  |  65.44   | 48.63 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-clip_bit_512x512_80k_s2looking.py) |
|  BAN   | ViT-L/14, CLIP | ChangeFormer-b0 |  512x512  |  80000  |   74.63   | 60.30  |  66.70   | 50.04 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ban/ban_vit-l14-clip_mit-b0_512x512_80k_s2looking.py) |


- You can download pretrained ViTs from [huggingface](https://huggingface.co/likyoo/BAN/tree/main/pretrain) | [baidu disk](https://pan.baidu.com/s/1RkIGsOB3XBi7Oi6mKIpZ2w?pwd=kfp9) for training.
- You can download checkpoint files from [huggingface](https://huggingface.co/likyoo/BAN/tree/main/checkpoint) | [baidu disk](https://pan.baidu.com/s/1RkIGsOB3XBi7Oi6mKIpZ2w?pwd=kfp9) for evaluation.
- All metrics are based on the category "change".
- All scores are computed on the test set.
