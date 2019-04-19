# Installing Nuke Deep Learning Plug-in

The Nuke Deep Learning (DL) installation can be divided into compiling the DLClient Nuke node and installing the DLServer using Docker.

**Requirements:**
- NVIDIA GPU, Linux with Nuke installed
- CMake (minimum 3.10)
- Protobuf (tested with 2.5.0 and 3.5.1)
- Docker

## Installing the client

### Install Protobuf

Protobuf may be installed with a package manager, for example:
```
sudo yum install protobuf-devel
```

However we recommend compiling it from source following the [installation instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) from the protobuf github repository.

Get Protobuf source file for C++, for instance version 3.5.1:
```
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz
# Extract file in current directory
tar -xzf protobuf-cpp-3.5.1.tar.gz
```
To build and install the C++ Protocol Buffer runtime and the Protocol Buffer compiler (protoc), execute the following:
```
cd protobuf-3.5.1
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```
Update LD_LIBRARY_PATH to point to protobuf:
```
export LD_LIBRARY_PATH=/path/to/protobuf/:$LD_LIBRARY_PATH
```

### Compile DLClient Nuke node

If not already cloned, fetch the `nuke-DL-server` repository:
```
git clone https://github.com/TheFoundryVisionmongers/nuke-DL-server
```
Execute the commands below to compile the client DLClient.so plug-in, setting the NUKE_INSTALL_PATH to point to the folder of the desired Nuke version:
```
mkdir build && cd build
cmake -DNUKE_INSTALL_PATH="/path/to/Nuke11.3v1" ..
make
```
The DLClient.so plug-in will now be in the 'build/Plugins/Client' folder. Before it can be used, Nuke needs to know where it lives. One way to do this is to update the NUKE_PATH environment variable to point to the DLClient.so plug-in (This can be skipped if it was moved to the root of your ~/.nuke folder, or the path was added in Nuke through Python):
```
export NUKE_PATH=/path/to/lib/:$NUKE_PATH
```
At that point, after opening Nuke and doing an "Update [All plugins]", the "DLClient" node should be available.
If not verify that the NUKE_PATH is correctly set in this instance of Nuke (or simply export the NUKE_PATH in the ~/.bashrc)

## Installing the server

### Docker

Install Docker:
```
# Install Docker
sudo curl -sSL https://get.docker.com/ | sh
# Start Docker
sudo systemctl start docker
```
Install nvidia-docker (NVIDIA GPU-enabled docker) for your Linux platform by following the [installation instructions](https://github.com/NVIDIA/nvidia-docker) of the nvidia-docker repository (For CentOS/RHEL, follow the instructions using the official `docker-ce` package).

Build the docker image from the [Dockerfile](/Plugins/Server/Dockerfile):
```
# Start by loading Ubuntu16.04 with cuda 9.0 and cudnn7 as the base image
sudo docker pull nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Build the docker image on top of the base image
cd Plugins/Server/
# Choose your own label for <docker_image_name>, it must be lowercase. e.g. dlserver.
sudo docker build -t <docker_image_name> -f Dockerfile .
```

Create a docker container on top of the created docker image, referencing the `<docker_image_name>` from the previous step:

```
sudo nvidia-docker run -v /absolute/path/to/nuke-DL-server/Models/:/workspace/dl-server/models:ro -it <docker_image_name>
```

Note: the `-v` (volume) options links your nuke-DL-server/Models/ folder with the models/ folder inside your container. You only need to modify `/absolute/path/to/nuke-DL-server/Models/`, leave the `/workspace/dl-server/models:ro` unchanged as it already corresponds to the folder structure inside your Docker image. This option allows you to add models in Models/ that will be directly available and updated inside your container.

If you get
```
/bin/nvidia-docker: line 34: /bin/docker: Permission denied
/bin/nvidia-docker: line 34: /bin/docker: Success
```
try to replace the previous command with:
```
sudo docker run --runtime=nvidia -v /absolute/path/to/nuke-DL-server/Models/:/workspace/dl-server/models:ro -it <docker_image_name>
```

## Getting started

### Download configuration and weights files

To be able to run inference on both Densepose and Mask-RCNN deep learning models, you need to download the configuration and weight files:
- For Mask-RCNN:
  - Configuration: [e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml](https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml)
  - Correponding weights: [model_final.pkl](https://dl.fbaipublicfiles.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) (from the Detectron [Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md))
- For DensePose:
  - Configuration: [DensePose_ResNet101_FPN_s1x-e2e.yaml](https://github.com/facebookresearch/DensePose/blob/master/configs/DensePose_ResNet101_FPN_s1x-e2e.yaml)
  - Corresponding weights: [DensePose_ResNet101_FPN_s1x-e2e.pkl](https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl) (from the Densepose [Model Zoo](https://github.com/facebookresearch/DensePose/blob/master/MODEL_ZOO.md))

And respectively move them to Models/mrcnn/ and Models/densepose/ folders.

### Connect client and server

Finally to connect the Python server with the Nuke client:
1. In the running docker container, query the ip address:
```
hostname -I
```
2. In Nuke, set the DLClient node host to the container ip address,
3. In the container, launch the server and start listening on port 55555:
```
python server.py 55555
```
4. In Nuke, click on the DLClient connect button, you should have the three models available.

### Add your own model

To implement your own model, you can create a new folder in the /Models directory with your model name. At the minimum, this folder needs to include an empty `__init__.py` file and a `model.py` file that contains a Model class inheriting from BaseModel.

You can copy the simple [Models/blur/](Models/blur) model as a starting point, and implement your own model looking at the examples of blur, densepose and mrcnn.
