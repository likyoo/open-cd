# ChangeFormer

[A Transformer-Based Siamese Network for Change Detection](https://arxiv.org/abs/2201.01293)

## Introduction



## Abstract
This paper presents a transformer-based Siamese network architecture (abbreviated by ChangeFormer) for Change Detection (CD) from a pair of co-registered remote sensing images. Different from recent CD frameworks, which are based on fully convolutional networks (ConvNets), the proposed method unifies hierarchically structured transformer encoder with Multi-Layer Perception (MLP) decoder in a Siamese network architecture to efficiently render multi-scale long-range details required for accurate CD. Experiments on two CD datasets show that the proposed end-to-end trainable ChangeFormer architecture achieves better CD performance than previous counterparts. Our code is available at [this https URL](https://github.com/wgcban/ChangeFormer)

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/210196188-08eea25e-90e8-4ea9-921b-d63ef7a21274.png" width="90%"/>
</div>

```bibtex
@misc{bandara2022transformerbased,
      title={A Transformer-Based Siamese Network for Change Detection}, 
      author={Wele Gedara Chaminda Bandara and Vishal M. Patel},
      year={2022},
      eprint={2201.01293},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Results and models

### LEVIR-CD

|    Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :----------: | -------- | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| ChangeFormer | MIT-B0   |  256x256  |  40000  |    -     |   93.00   | 88.17  |  90.52   | 82.68 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py) |          |
| ChangeFormer | MIT-B1   |  256x256  |  40000  |    -     |   92.59   | 89.68  |  91.11   | 83.67 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changeformer/changeformer_mit-b1_256x256_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.
- We simply convert the Segformer to a siamese variant and do not strictly refer to the ChangeFormer.
