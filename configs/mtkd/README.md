

**[JL1-CD: A New Benchmark for Remote Sensing Change Detection and a Robust Multi-Teacher Knowledge Distillation Framework](https://arxiv.org/pdf/2502.13407)**


## Dataset
The JL1-CD dataset is now publicly available. You can download the checkpoint files from:

- [Google Drive](https://drive.google.com/drive/folders/1ELoqx7J3GrEFMX5_rRynMjW9-Poxz3Uu?usp=sharing)
- [Baidu Disk](https://pan.baidu.com/s/1_vcO4c5DM5LDuOqLwLrWJg?pwd=5byn)
- [Hugging Face](https://huggingface.co/datasets/circleLZY/JL1-CD)
## Usage

### Training

The training process for the MTKD framework consists of three steps. Below, we use the **Changer-MiT-b0** model as an example:

#### Step 1: Train the original model

Run the following command to train the original model:

```bash
python tools/train.py --work-dir --work-dir /path/to/save/models/Changer-mit-b0/initial
```

#### Step 2: Train teacher models for different CAR partitions (e.g., 3 partitions)

Train the teacher models for small, medium, and large CAR partitions as follows:

```bash
python tools/train.py configs/mtkd/step2/small-changer_ex_mit-b0_512x512_200k_jl1cd.py --work-dir /path/to/save/models/Changer-mit-b0/small

python tools/train.py configs/mtkd/step2/medium-changer_ex_mit-b0_512x512_200k_jl1cd.py --work-dir /path/to/save/models/Changer-mit-b0/medium

python tools/train.py configs/mtkd/step2/large-changer_ex_mit-b0_512x512_200k_jl1cd.py --work-dir /path/to/save/models/Changer-mit-b0/large
```

In the above two steps, you will have four model versions for **Changer-MiT-b0**: the original model and three teacher models (small, medium, and large). At this point, the O-P strategy can already be applied.

#### Step 3: Train the student model

Initialize the checkpoint paths in `configs/mtkd/step3/mtkd-changer_ex_mit-b0_512x512_200k_jl1cd.py` for the student model and teacher models as follows:

- `checkpoint_student`
- `checkpoint_teacher_l`
- `checkpoint_teacher_m`
- `checkpoint_teacher_s`

Then, run the following command to train the student model:

```bash
python tools/train.py configs/mtkd/step3/mtkd-changer_ex_mit-b0_512x512_200k_jl1cd.py --work-dir /path/to/save/models/Changer-mit-b0/distill
```

After this step, you will have the student model trained within the MTKD framework.

### Testing

Testing the student model trained with MTKD is simple. Run the following command:

```bash
python test.py <config-file> <checkpoint>
```

Testing the O-P strategy is more complex. You can refer to the script located at `tools/test_pipline/single-partition-3-test.py` for more details.

#### Checkpoints

You can download checkpoint files from:
- [Baidu Disk](https://pan.baidu.com/s/1F5MIGCCiNHFifNl_kDiklA?pwd=4tid)
- [Hugging Face](https://huggingface.co/circleLZY/MTKD)


## Citation

If you find the JL1-CD dataset or our work useful in your research, please consider citing our paper:

```bibtex
@article{liu2025jl1,
  title={JL1-CD: A New Benchmark for Remote Sensing Change Detection and a Robust Multi-Teacher Knowledge Distillation Framework},
  author={Liu, Ziyuan and Zhu, Ruifei and Gao, Long and Zhou, Yuanxiu and Ma, Jingyu and Gu, Yuantao},
  journal={arXiv preprint arXiv:2502.13407},
  year={2025}
}
```
