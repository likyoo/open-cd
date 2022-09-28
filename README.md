<div align="center">
  <img src="resources/opencd-logo.png" width="600"/>
</div>

## Introduction
Open-CD is an open source change detection toolbox based on a series of open source general vision task tools.


## News
- 9/28/2022 - The code, pre-trained models and logs of [ChangerEx](https://github.com/likyoo/open-cd/tree/main/configs/changer) are available. :yum:
- 9/20/2022 - Our paper [Changer: Feature Interaction is What You Need for Change Detection](https://arxiv.org/abs/2209.08290) is available!
- 7/30/2022 - Open-CD is publicly available!

## Plan

Support for

- [x] [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [ ] ...


## Usage

[Docs](https://github.com/open-mmlab/mmsegmentation/tree/master/docs)

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) in mmseg.

#### simple usage
```
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
```
train
```
python tools/train.py configs/changer/changer_r18_512x512_40k_levircd.py --work-dir ./changer_r18_levir_workdir --gpu-id 0 --seed 307
```
infer
```
# get .png results
python tools/test.py configs/changer/changer_r18_512x512_40k_levircd.py  changer_r18_levir_workdir/latest.pth --format-only --eval-options "imgfile_prefix=tmp_infer"
# get metrics
python tools/test.py configs/changer/changer_r18_512x512_40k_levircd.py  changer_r18_levir_workdir/latest.pth --eval mFscore
```

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{fang2022changer,
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  author={Sheng Fang and Kaiyu Li and Zhe Li},
  year={2022},
  eprint={2209.08290},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{opencd2022,
    title={{Open-CD}: An open source change detection toolbox},
    author={Open-CD Contributors},
    howpublished = {\url{https://github.com/likyoo/open-cd}},
    year={2022}
}
```

## License

Open-CD is released under the Apache 2.0 license.


----
**The inspiration for this project comes from a casual conversation with friends during my internship at Sensetime.**

