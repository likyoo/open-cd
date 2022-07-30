<div align="center">
  <img src="resources/opencd-logo.png" width="600"/>
</div>

## Introduction
Open-CD is an open source change detection toolbox based on a series of open source general vision task tools.


## News
- 7/30/2022 - Our related paper is coming soon!
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
python tools/train.py configs/upernet/upernet_r18_512x512_20k_levircd.py --work-dir ./tmp_work_dir --gpu-id 0 --seed 307
```
infer
```
# get .png results
python tools/test.py configs/upernet/upernet_r50_512x512_20k_levircd.py  tmp_work_dir/latest.pth --format-only --eval-options "imgfile_prefix=tmp_infer"
# get metrics
python tools/test.py configs/upernet/upernet_r50_512x512_20k_levircd.py  tmp_work_dir/latest.pth --eval mFscore
```

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{opencd2022,
    title={{Open-CD}: An open source change detection toolbox},
    author={Open-CD Contributors},
    howpublished = {\url{https://github.com/likyoo/open-cd}},
    year={2022}
}
```

## License

Open-CD is released under the Apache 2.0 license, while some specific features in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

----
**The inspiration for this project comes from a casual conversation with friends during my internship at Sensetime.**

