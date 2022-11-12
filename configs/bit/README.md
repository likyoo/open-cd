# BiT

[Remote Sensing Image Change Detection with Transformers](https://arxiv.org/abs/2103.00208)

## Introduction

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/bit_head.py)

## Abstract

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

