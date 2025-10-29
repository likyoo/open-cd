<div align="center">
  <img src="resources/opencd-logo.png" width="600"/>
</div>

------

<font size=4><div align='center' > 
[[🤗Paper](https://arxiv.org/abs/2407.15317)] | [[🤗Code](https://github.com/likyoo/open-cd)] | [[🤗Checkpoint](https://huggingface.co/likyoo/Open-CD_Model_Zoo)] | [[🤗Video](https://www.youtube.com/watch?v=6Yt9g-vS_FU&t=4s)]</div></font>

## Introduction
Open-CD is an open source change detection toolbox based on a series of open source general vision task tools.


## News
- 7/25/2025 - Open-CD Technical Report is accepted by ACMMM 2025. 🎉
- 4/11/2025 - [MTKD](https://github.com/circleLZY/MTKD-CD) method and [JL1-CD](https://github.com/circleLZY/MTKD-CD) dataset are supported. Open-CD Technical Report is updated to v1.1.
- 7/23/2024 - **Open-CD Technical Report v1.0 is released on [arXiv](https://arxiv.org/abs/2407.15317), thanks to all contributors! Feel free to join us!** 💥💥💥
- 6/29/2024 - [ChangeStar](https://github.com/Z-Zheng/ChangeStar) and [FarSeg](https://github.com/Z-Zheng/FarSeg) are supported.
- 6/20/2024 - We launch the **[Open-CD Technical Report Plan](https://github.com/likyoo/open-cd/tree/main/projects/open-cd_technical_report)**, don't hesitate to join us!!! 💥💥💥
- 6/17/2024 - [CGNet](https://github.com/ChengxiHAN/CGNet-CD) is supported.
- 2/10/2024 - Open-CD is upgraded to v1.1.0. [BAN](https://github.com/likyoo/BAN), [TTP](https://github.com/KyanChen/TTP) and [LightCDNet](https://github.com/NightSongs/LightCDNet) is supported. The inference API is added.
- 4/21/2023 - Open-CD v1.0.0 is released in 1.x branch, based on OpenMMLab 2.0 ! PyTorch 2.0 is also supported ! Enjoy it !
- 3/14/2023 - Open-CD is upgraded to v0.0.3. Semantic Change Detection (SCD) is supported !
- 11/17/2022 - Open-CD is upgraded to v0.0.2, requiring a higher version of the MMSegmentation dependency.
- 9/28/2022 - The code, pre-trained models and logs of [ChangerEx](https://github.com/likyoo/open-cd/tree/main/configs/changer) are available. :yum:
- 9/20/2022 - Our paper [Changer: Feature Interaction is What You Need for Change Detection](https://arxiv.org/abs/2209.08290) is available!
- 7/30/2022 - Open-CD is publicly available!

## Benchmark and model zoo

Supported toolboxes:

- [x] [OpenMMLab Toolkits](https://github.com/open-mmlab)
- [x] [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [ ] ...

Supported change detection model:
(_The code of some models are borrowed directly from their official repositories._)

- [x] [FC-EF (ICIP'2018)](configs/fcsn)
- [x] [FC-Siam-diff (ICIP'2018)](configs/fcsn)
- [x] [FC-Siam-conc (ICIP'2018)](configs/fcsn)
- [x] [STANet (RS'2020)](configs/stanet)
- [x] [IFN (ISPRS'2020)](configs/ifn)
- [x] [SNUNet (GRSL'2021)](configs/snunet)
- [x] [BiT (TGRS'2021)](configs/bit)
- [x] [ChangeStar (ICCV'2021)](configs/changestar)
- [x] [ChangeFormer (IGARSS'22)](configs/changeformer)
- [x] [TinyCD (NCA'2023)](configs/tinycd)
- [x] [Changer (TGRS'2023)](configs/changer)
- [x] [HANet (JSTARS'2023)](configs/hanet)
- [x] [TinyCDv2 (Under Review)](configs/tinycd_v2)
- [x] [LightCDNet (GRSL'2023)](configs/lightcdnet)
- [x] [CGNet (JSTARS'2023)](configs/cgnet)
- [x] [BAN (TGRS'2024)](configs/ban)
- [x] [TTP (arXiv'2023)](configs/ttp)
- [x] [MTKD (arXiv'2025)](configs/mtkd)
- [ ] ...

Supported datasets: | [Descriptions](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
- [x] [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [x] [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
- [x] [S2Looking](https://github.com/S2Looking/Dataset)
- [x] [SVCD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- [x] [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)
- [x] [CLCD](https://github.com/liumency/CropLand-CD)
- [x] [RSIPAC](https://engine.piesat.cn/ai/autolearning/index.html#/dataset/detail?key=8f6c7645-e60f-42ce-9af3-2c66e95cfa27)
- [x] [SECOND](https://captain-whu.github.io/SCD/)
- [x] [Landsat](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)
- [x] [BANDON](https://github.com/fitzpchao/BANDON)
- [x] [JL1-CD](https://github.com/circleLZY/MTKD-CD)
- [ ] ...

## Usage

[Docs](https://github.com/open-mmlab/mmsegmentation/tree/master/docs)

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) in mmseg.

A Colab tutorial is also provided. You may directly run on [Colab](https://colab.research.google.com/drive/1puZY5R8fwlL6um6pHbgbM1NTYZUXdK2J?usp=sharing). (thanks to [@Agustin](https://github.com/AgustinNormand) for this demo) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1puZY5R8fwlL6um6pHbgbM1NTYZUXdK2J?usp=sharing)

#### Install

```
# Install OpenMMLab Toolkits as Python packages
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.2.2"
pip install "mmdet>=3.0.0"
```
```
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
```
For more details, please see [here](https://github.com/likyoo/open-cd/blob/main/docs/install.md).

#### Train
```
python tools/train.py configs/changer/changer_ex_r18_512x512_40k_levircd.py --work-dir ./changer_r18_levir_workdir
```

#### Test
```
# get .png results
python tools/test.py configs/changer/changer_ex_r18_512x512_40k_levircd.py changer_r18_levir_workdir/latest.pth --show-dir tmp_infer
# get metrics
python tools/test.py configs/changer/changer_ex_r18_512x512_40k_levircd.py changer_r18_levir_workdir/latest.pth
```

#### Infer
Please refer [inference](https://github.com/likyoo/open-cd/blob/main/docs/inference.md) doc.


## Citation

If you find this project useful in your research, please cite:

```bibtex
@article{opencd,
  title   = {{Open-CD}: A Comprehensive Toolbox for Change Detection},
  author  = {Li, Kaiyu and Jiang, Jiawei and Codegoni, Andrea and Han, Chengxi and Deng, Yupeng and Chen, Keyan and Zheng, Zhuo and
             Chen, Hao and Liu, Ziyuan and Gu, Yuantao and Zou, Zhengxia and Shi, Zhenwei and Fang, Sheng and Meng, Deyu and Wang, Zhi and Cao, Xiangyong},
  journal= {arXiv preprint arXiv:2407.15317},
  year={2024}
}
```
You might also consider citing:

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

@ARTICLE{10129139,
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2023.3277496}}
```

## License

Open-CD is released under the Apache 2.0 license.
