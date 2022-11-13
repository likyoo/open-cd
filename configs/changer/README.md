# Changer

[Changer: Feature Interaction is What You Need for Change Detection](https://arxiv.org/abs/2209.08290)

## Introduction

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/changer.py#L151)

## Abstract
Change detection is an important tool for long-term earth observation missions. It takes bi-temporal images as input and predicts “where” the change has occurred. Different from other dense prediction tasks, a meaningful consideration for change detection is the interaction between bi-temporal features. With this motivation, in this paper we propose a novel general change detection architecture, MetaChanger, which includes a series of alternative interaction layers in the feature extractor. To verify the effectiveness of MetaChanger, we propose two derived models, ChangerAD and ChangerEx with simple interaction strategies: Aggregation-Distribution (AD) and “exchange”. AD is abstracted from some complex interaction methods, and “exchange” is a completely parameter&computation-free operation by exchanging bi-temporal features. In addition, for better alignment of bi-temporal features, we propose a Flow Dual-Alignment Fusion (FDAF) module which allows interactive alignment and feature fusion. Crucially, we observe Changer series models achieve competitive performance on different scale change detection datasets. Further, our proposed ChangerAD and ChangerEx could serve as a starting baseline for future MetaChanger design.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/192922229-9a9480c2-cb12-42e5-84e6-92ee1df1f775.png" width="90%"/>
</div>

```bibtex
@article{fang2022changer,
  title={Changer: Feature Interaction is What You Need for Change Detection},
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={arXiv preprint arXiv:2209.08290},
  year={2022}
}
```

## Results and models

### S2Looking

|  Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            |                           download                           |
| :-------: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ChangerEx |   r18    |  512x512  |  80000  |    -     |   75.04   | 59.35  |  66.28   | 49.57 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changer/changer_ex_r18_512x512_80k_s2looking.py) | [model](https://drive.google.com/file/d/1yR-ORxgj7Hjm-J83Htv0rQiNc6w8x1Gs/view?usp=sharing) \| [log](https://drive.google.com/file/d/1JWGh_N7GqSi9kodbK_MhRhYON7suJG7u/view?usp=sharing) |
| ChangerEx |   s50    |  512x512  |  80000  |    -     |   74.63   | 61.08  |  67.18   | 50.58 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changer/changer_ex_s50_512x512_80k_s2looking.py) | [model](https://drive.google.com/file/d/1JLZ95FJD32zpTAT3BBvy4YnzFuaC_wXF/view?usp=sharing) \| [log](https://drive.google.com/file/d/1XpMIyVbrFZOVxeMWk3pGIOz6X92UDvzz/view?usp=sharing) |
| ChangerEx |   s101   |  512x512  |  80000  |    -     |   74.40   | 61.95  |  67.61   | 51.07 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changer/changer_ex_s101_512x512_80k_s2looking.py) | [model](https://drive.google.com/file/d/1PevW2rQZILEmPyW6YRxu3D33-GkT1YJ7/view?usp=sharing) \| [log](https://drive.google.com/file/d/1bIaxr-bbKSEyCHg6mo05zoKOm8rXARps/view?usp=sharing) |



### LEVIR-CD

|  Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            |                           download                           |
| :-------: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ChangerEx |   r18    |  512x512  |  40000  |    -     |   92.97   | 90.61  |  91.77   | 84.80 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changer/changer_ex_r18_512x512_40k_levircd.py) |                      lost​ :confounded:                       |
| ChangerEx |   s50    |  512x512  |  40000  |    -     |   93.47   | 90.95  |  92.19   | 85.51 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changer/changer_ex_s50_512x512_40k_levircd.py) | [model](https://drive.google.com/file/d/1rnQjWrMShB2bHOjMqARyhGn0p-nQFiaB/view?usp=sharing) \| [log](https://drive.google.com/file/d/1sRyKvVBJghjPRjq4cqt_qYdSbxxJIIsj/view?usp=sharing) |
| ChangerEx |   s101   |  512x512  |  40000  |    -     |   93.38   | 91.31  |  92.33   | 85.76 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changer/changer_ex_s101_512x512_40k_levircd.py) | [model](https://drive.google.com/file/d/128FVQL-93oN5lUMGuqDcmPU-80RiXBhn/view?usp=sharing) \| [log](https://drive.google.com/file/d/18qcXeyC6rq-l04vS5I5n3VAhh0HsCxFF/view?usp=sharing) |


- All metrics are based on the category "change".
- All scores are computed on the test set.
