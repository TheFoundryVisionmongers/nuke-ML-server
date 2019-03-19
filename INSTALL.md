# Installing Nuke Deep Learning Plug-in

The Nuke Deep Learning (DL) installation can be divided into compiling the DLClient Nuke node and installing the DLServer using Docker.

**Requirements:**
- NVIDIA GPU, Linux with Nuke installed (XXX Nuke11.3v1 required or other version fine as well?)
- Protobuf
- Docker

## Installing the Client

### Install Protobuf

Following the [installation instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) from the protobuf github repository.

Get Protobuf 3.5.1 source file for C++:
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

Clone the nuke-dl-server repository:
```
# NUKEDLSERVER=/path/to/clone/nuke-DL-server
git clone https://github.com/TheFoundryVisionmongers/nuke-DL-server $NUKEDLSERVER
```
Compile the Nuke node and make the DLClient.so plug-in library:
```
mkdir build && cd build
cmake ..
make
```
Update NUKE_PATH to point to the shared DLClient.so library:
```
export NUKE_PATH=/path/to/build/lib/:$NUKE_PATH
```
At that point, after opening Nuke and doing an "Update [All plugins]", the "DLClient" node should be available.
If not verify that the NUKE_PATH is correctly set in this instance of Nuke (or simply export the NUKE_PATH in the ~/.bashrc)

## Installing the Server

### Docker

Install Docker:
```
# Install Docker
sudo curl -sSL https://get.docker.com/ | sh
# Start Docker
sudo systemctl start docker
```
Install nvidia-docker (NVIDIA GPU-enabled docker) for your Linux platform by following the [installation instructions](https://github.com/NVIDIA/nvidia-docker) of the nvidia-docker repository.

Build the docker image from the [Dockerfile](/Plugins/Server/Dockerfile):
```
# Start by loading Ubuntu16.04 with cuda 9.0 and cudnn7 as the base image
sudo docker pull nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Next line necessary for now to correctly copy the Server directory to the docker dl-server directory (XXX change Dockerfile to directly clone from github)
cd Plugins/Server/
# Build the docker image on top of the base image
sudo docker build -t <docker_image_name> -f Dockerfile .
```
Create a docker container on top of the created docker image:
```
sudo nvidia-docker run -v /path/to/Models/:/workspace/dl-server/models:ro --name <container_name> -it <docker_image_name>
```
Note: the `-v` (volume) options links your Models/ folder with the models/ folder inside your container. Thus you can add models in Models/ that will be directly available and updated inside your container.

## Getting started

### Download configuration and weights files

To be able to run inference on both Densepose and Mask-RCNN deep learning models, you need to download the configuration and weight files:
- For Mask-RCNN:
  - Configuration: [e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml](https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml),
  - Correponding weights: [model_final.pkl](https://dl.fbaipublicfiles.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) (from the Detectron [Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md))
- For DensePose:
  - Configuration: [DensePose_ResNet101_FPN_s1x-e2e.yaml](https://github.com/facebookresearch/DensePose/blob/master/configs/DensePose_ResNet101_FPN_s1x-e2e.yaml)
  - Corresponding weights: [DensePose_ResNet101_FPN_s1x-e2e.pkl](https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl) (from the Densepose [Model Zoo](https://github.com/facebookresearch/DensePose/blob/master/MODEL_ZOO.md))

And respectively move them to Models/mrcnn and Models/densePose.

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