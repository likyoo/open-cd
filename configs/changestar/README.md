# ChangeStar

[Change is Everywhere Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery](https://arxiv.org/abs/2108.07002)

## Introduction

[Official Repo](https://github.com/Z-Zheng/ChangeStar)

[Code Snippet](https://github.com/likyoo/open-cd/blob/main/opencd/models/decode_heads/changerstar_head.py)

## Abstract
For high spatial resolution (HSR) remote sensing images, bitemporal supervised learning always dominates change detection using many pairwise labeled bitemporal images. However, it is very expensive and time-consuming to pairwise label large-scale bitemporal HSR remote sensing images. In this paper, we propose single-temporal supervised learning (STAR) for change detection from a new perspective of exploiting object changes in unpaired images as supervisory signals. STAR enables us to train a high-accuracy change detector only using unpaired labeled images and generalize to real-world bitemporal images. To evaluate the effectiveness of STAR, we design a simple yet effective change detector called ChangeStar, which can reuse any deep semantic segmentation architecture by the ChangeMixin module. The comprehensive experimental results show that ChangeStar outperforms the baseline with a large margin under single-temporal supervision and achieves superior performance under bitemporal supervision. Code is available at https://github.com/Z-Zheng/ChangeStar.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/changestar.png" width="90%"/>
</div>


```bibtex
@inproceedings{zheng2021change,
  title={Change is everywhere: Single-temporal supervised object change detection in remote sensing imagery},
  author={Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={15193--15202},
  year={2021}
}
```

## Results and models

### LEVIR-CD

|   Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            | download |
| :--------: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :------: |
| ChangeStar |   r18    |  512x512  |  40000  |    -     |   93.75    | 88.90  |  91.26   | 83.92 | [config](https://github.com/likyoo/open-cd/blob/main/configs/changestar/changestar_farseg_1x96_512x512_40k_levircd.py) |          |


- All metrics are based on the category "change".
- All scores are computed on the test set.

