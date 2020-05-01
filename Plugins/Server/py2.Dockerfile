# Ubuntu 18.04 with CUDA 10.0, CuDNN 7.6
# Python2.7, TensorFlow 1.14.0, PyTorch 1.0
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

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

# Install Python 2.7
RUN apt-get update && apt-get install -y --no-install-recommends \
        python-pip \
        python2.7-dev && \
    rm -rf /var/lib/apt/lists/*

# pip version 21.0 will drop support for Python 2.7
RUN python -m pip install --upgrade pip==20.1
RUN pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir \
    future \
    gast==0.2.2 \
    protobuf \
    pyyaml==3.13 \
    scikit-image \
    typing \
    imageio==2.6.1 \
    OpenEXR==1.3.2

# Install TF 1.15.0 GPU for Python2.7
RUN pip install --no-cache-dir \
    tensorflow-gpu==1.15.0 \
    tensorflow-determinism

# Install PyTorch (include Caffe2) for CUDA 10.0
RUN pip install --no-cache-dir torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir cupy-cuda100
RUN pip install --no-cache-dir cython

WORKDIR /workspace
# Install the COCO API
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron for mask RCNN
RUN git clone https://github.com/facebookresearch/detectron
RUN cd detectron && pip install -r requirements.txt && make

WORKDIR /workspace/ml-server
# Copy your current folder to the docker image /workspace/ml-server/ folder
COPY . .