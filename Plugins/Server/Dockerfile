# Ubuntu 18.04 with CUDA 10.0, CuDNN 7.6
# Python3.6, TensorFlow 1.15.0, PyTorch 1.4
FROM nvcr.io/nvidia/tensorflow:23.02-tf1-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libglib2.0-0 \
        libjpeg-dev \
        libopencv-dev \
        libopenexr-dev \
        libpng-dev \
        libsm-dev \
        vim && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.6
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        python3-opencv \
#        python3-pip \
#        python3.9-dev python3.9-venv && \
#    rm -rf /var/lib/apt/lists/*
# Have aliases python3->python and pip3->pip
#RUN unlink /usr/bin/python && \
#    unlink /usr/bin/pip && \
#    ln -s /usr/bin/python3.9 /usr/bin/python && \
#    ln -s /usr/bin/pip3 /usr/bin/pip
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir \
    future \
    gast==0.2.2 \
    protobuf \
    pyyaml==3.13 \
    scikit-image \
    typing \
    imageio \
    OpenEXR
# Install TF 1.15.0 GPU for Python3.6 (no TensorRT)
#RUN pip install --no-cache-dir \
#    tensorflow-gpu==1.15.0 \
#    tensorflow-determinism

RUN python -m pip install nvidia-pyindex --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com 
RUN python -m pip install  nvidia-tensorflow[horovod] --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com 

# Install PyTorch (include Caffe2) for CUDA 10.0
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# CPU only
RUN pip install --no-cache-dir cupy-cuda12x
RUN pip install --no-cache-dir cython

WORKDIR /workspace
# Install the COCO API
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron for mask RCNN
#RUN pip install 'git+https://github.com/johnnynunez/detectron2.git'
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2
WORKDIR /detectron2
RUN pip install -e .

WORKDIR /workspace/ml-server
# Copy your current folder to the docker image /workspace/ml-server/ folder
COPY . .
