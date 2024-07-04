### For example

1. Create Conda environment

    ```
    conda create -n opencd python=3.8
    ```

2. Install PyTorch

    We strongly recommend directly installing the version of [PyTorch](https://pytorch.org/get-started/previous-versions/) for which a pre-built version of mmcv exists. 

    A detailed list can be found [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip).

    ```
    # install pytorch 1.13.1
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

3. Install OpenMMLab Toolkits

    ```
    # Install OpenMMLab Toolkits as Python packages
    pip install -U openmim
    mim install mmengine==0.10.4
    mim install mmcv==2.1.0
    mim install mmpretrain==1.2.0
    pip install mmsegmentation==1.2.2
    pip install mmdet==3.3.0
    ```

    ```
    cd open-cd
    pip install -v -e .
    ```

4. Install other dependencies

    ```
    pip install ftfy
    pip install regex
    ```