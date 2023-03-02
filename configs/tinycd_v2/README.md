# TinyCDv2

TinyCDv2: A Lighter and Stronger Network for Remote Sensing Change Detection

## Introduction

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/tinynet.py)

## Abstract


### LEVIR-CD

|   Method   | Crop Size | Lr schd | #Param (M) | MACs (G) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :--------: | :-------: | :-----: | :--------: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| TinyCDv2-S |  256x256  |  40000  |   0.045    |   0.23   |   90.08   | 89.44  |  89.76   | 81.42 | [config](https://github.com/likyoo/open-cd/blob/main/configs/tinycd_v2/tinycd_v2_s_256x256_40k_levircd.py) |          |
| TinyCDv2-B |  256x256  |  40000  |   0.116    |   0.58   |   91.75   | 90.17  |  90.95   | 83.41 | [config](https://github.com/likyoo/open-cd/blob/main/configs/tinycd_v2/tinycd_v2_b_256x256_40k_levircd.py) |          |
| TinyCDv2-L |  256x256  |  40000  |   0.263    |   1.51   |   92.27   | 90.48  |  91.37   | 84.11 | [config](https://github.com/likyoo/open-cd/blob/main/configs/tinycd_v2/tinycd_v2_l_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
