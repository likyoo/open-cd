# TTP

[Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection](https://arxiv.org/abs/2312.16202)

## Introduction

[Official Repo](https://github.com/KyanChen/TTP)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/change_detectors/ttp.py)



## Abstract

Change detection, a prominent research area in remote sensing, is pivotal in observing and analyzing surface transformations. Despite significant advancements achieved through deep learning-based methods, executing high-precision change detection in spatio-temporally complex remote sensing scenarios still presents a substantial challenge. The recent emergence of foundation models, with their powerful universality and generalization capabilities, offers potential solutions. However, bridging the gap of data and tasks remains a significant obstacle. In this paper, we introduce Time Travelling Pixels (TTP), a novel approach that integrates the latent knowledge of the SAM foundation model into change detection. This method effectively addresses the domain shift in general knowledge transfer and the challenge of expressing homogeneous and heterogeneous characteristics of multi-temporal images. The state-of-the-art results obtained on the LEVIR-CD underscore the efficacy of the TTP. The Code is available at https://kychen.me/TTP.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/likyoo/open-cd/assets/44317497/a61cd241-f9fb-4fd8-82c8-a20633222db5" width="100%"/>
</div>


```bibtex
@article{chen2023time,
  title={Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection},
  author={Chen, Keyan and Liu, Chengyang and Li, Wenyuan and Liu, Zili and Chen, Hao and Zhang, Haotian and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2312.16202},
  year={2023}
}
```

## Dependencies 

```
pip install peft
```

## Results and models

### LEVIR-CD

| Method | Backbone  | Crop Size | Lr schd | Precision | Recall | F1-Score | IoU  |                            config                            |
| :----: | --------- | :-------: | :-----: | :-------: | :----: | :------: | :--: | :----------------------------------------------------------: |
|  TTP   | ViT-SAM-L |  512x512  |  300e   |   93.0    |  91.7  |   92.1   | 85.6 | [config](https://github.com/likyoo/open-cd/blob/main/configs/ttp/ttp_vit-sam-l_512x512_300e_levircd.py) |


- All metrics are based on the category "change".
- All scores are computed on the test set.
