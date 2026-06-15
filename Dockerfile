# slightly modified Dockerfile from mmsegmentation repo
# https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/docker/Dockerfile
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="2.0.1"

FROM docker.io/pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install openmim"]
RUN ["/bin/bash", "-c", "mim install mmengine"]
RUN ["/bin/bash", "-c", "mim install mmcv==${MMCV}"]

# Install MMSegmentation
RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

# ──────────────────────────────────────────────────────────────────────────────
# Install open-cd
# ──────────────────────────────────────────────────────────────────────────────
RUN ["/bin/bash", "-c", "pip install mmdet>=3.0.0"]
RUN ["/bin/bash", "-c", "mim install mmpretrain>=1.0.0"]
WORKDIR /workspace
COPY . /workspace
RUN pip install -e .

# Default command opens an interactive prompt
CMD ["bash"]
