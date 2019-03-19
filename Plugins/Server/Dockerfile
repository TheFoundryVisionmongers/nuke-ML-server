FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=2.7
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libglib2.0-0 \   
        libjpeg-dev \
        libopencv-dev \
        libpng-dev \
        libsm-dev \        
        vim && \
    rm -rf /var/lib/apt/lists/*

#install conda with miniconda, then install pytorch (which includes caffe2) and dependencies
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing
RUN /opt/conda/bin/conda install -y magma-cuda90 -c pytorch && \
    /opt/conda/bin/conda install -y pytorch-nightly -c pytorch && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip==9.0.3 setuptools wheel && \
    pip install --no-cache-dir \
    future \
    protobuf \
    scikit-image \
    tensorflow-gpu \
    cupy

WORKDIR /workspace

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && make install

WORKDIR /workspace
# Install the densepose library including detectron
RUN git clone https://github.com/facebookresearch/densepose
RUN cd densepose && pip install -r requirements.txt && make
# Next line to find lib/libcaffe2_detectron_ops_gpu.so 
RUN cp /opt/conda/lib/python2.7/site-packages/torch/lib/libcaffe2_detectron_ops_gpu.so /usr/local/lib

# Fetch DensePose UV data
RUN apt update && apt install wget
RUN bash /workspace/densepose/DensePoseData/get_densepose_uv.sh
RUN mv UV_data /workspace/densepose/DensePoseData/

WORKDIR /workspace/dl-server
# Copy your current folder to the docker image /workspace/dl-server/ folder
COPY . .