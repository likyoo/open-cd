# TinyCD

[TINYCD: A (Not So) Deep Learning Model For Change Detection](https://arxiv.org/abs/2207.13159)

## Introduction

[Official Repo](https://github.com/AndreaCodegoni/Tiny_model_4_CD)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/backbones/tinycd.py)

## Abstract
In this paper, we present a lightweight and effective change detection model, called TinyCD. This model has been designed to be faster and smaller than current state-of-the-art change detection models due to industrial needs. Despite being from 13 to 140 times smaller than the compared change detection models, and exposing at least a third of the computational complexity, our model outperforms the current state-of-the-art models by at least 1% on both F1 score and IoU on the LEVIR-CD dataset, and more than 8% on the WHU-CD dataset. To reach these results, TinyCD uses a Siamese U-Net architecture exploiting low-level features in a globally temporal and locally spatial way. In addition, it adopts a new strategy to mix features in the space-time domain both to merge the embeddings obtained from the Siamese backbones, and, coupled with an MLP block, it forms a novel space-semantic attention mechanism, the Mix and Attention Mask Block (MAMB). Source code, models and results are available here: [this https URL](https://github.com/AndreaCodegoni/Tiny_model_4_CD)

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/202379496-b88af0dc-fee4-40af-bef4-174d5fc26502.png" width="90%"/>
</div>

```bibtex
@article{codegoni2022tinycd,
  title={TINYCD: A (Not So) Deep Learning Model For Change Detection},
  author={Codegoni, Andrea and Lombardi, Gabriele and Ferrari, Alessandro},
  journal={arXiv preprint arXiv:2207.13159},
  year={2022}
}
```

## Results and models

### LEVIR-CD

| Method |   Backbone   | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----: | :----------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| TinyCD | EfficientNet |  256x256  |  40000  |          |   91.87   | 89.89  |  90.87   | 83.26 | [config](https://github.com/likyoo/open-cd/blob/main/configs/tinycd/tinycd_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
