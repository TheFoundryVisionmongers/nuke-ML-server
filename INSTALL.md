# Installing Nuke Machine Learning Plug-in

The Nuke Machine Learning (ML) installation can be divided into compiling the MLClient Nuke node and installing the MLServer using Docker.

**Requirements:**
- Linux with Nuke installed
- NVIDIA GPU (Important: GPU memory must be at least 6GB)
- CMake (minimum 3.10)
- Protobuf (tested with 2.5.0 and 3.5.1)
- Docker

## Installing the Client

### Install Protobuf

Protocol Buffers (aka Protobuf) are an efficient way of serializing structured data - similar to XML, but faster and simpler. We use it to define, write, and read the data for our client<->server communication.

Following the [installation instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) from the Protobuf GitHub repository, we recommend compiling Protobuf from source:

First get Protobuf source file for C++, for instance version 3.5.1:
```
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz
# Extract file in current directory
tar -xzf protobuf-cpp-3.5.1.tar.gz
```
Then build and install the C++ Protocol Buffer runtime and the Protocol Buffer compiler (protoc):
```
cd protobuf-3.5.1
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```

Note: Instead of compiling it from source, Protobuf may alternatively be installed with a package manager, for example:
```
sudo yum install protobuf-devel
```

### Compile MLClient Nuke Node

If not already cloned, fetch the `nuke-ML-server` repository:
```
git clone https://github.com/TheFoundryVisionmongers/nuke-ML-server
```
Execute the commands below to compile the client MLClient.so plug-in, setting the NUKE_INSTALL_PATH to point to the folder of the desired Nuke version:
```
cd nuke-ML-server/
mkdir build && cd build
cmake -DNUKE_INSTALL_PATH=/path/to/Nuke11.3v1/ ..
make
```
The MLClient.so plug-in will now be in the 'build/Plugins/Client' folder. Before it can be used, Nuke needs to know where it lives. One way to do this is to update the NUKE_PATH environment variable to point to the MLClient.so plug-in (This can be skipped if it was moved to the root of your ~/.nuke folder, or the path was added in Nuke through Python):
```
export NUKE_PATH=/path/to/lib/:$NUKE_PATH
```
At that point, after opening Nuke and updating all plugins, the `MLClient` node should be available. To update all the plugins in Nuke, you can either use the Other > All Plugins > Update option (see [documentation](https://learn.foundry.com/nuke/developers/63/pythondevguide/)installing_plugins.html)), or simply press `tab` in the Node Graph then write `Update [All plugins]`. If the `MLClient` node is still missing, verify that the current NUKE_PATH is correctly pointing to the folder containing MLClient.so.

## Installing the Server

### Install Docker

Docker provides a way to package and run an application in a securely isolated environment called a container. This container includes all the application dependencies and libraries. It ensures that the application works seamlessly inside the container in any system environment. We use docker to create a container that easily runs the MLServer.

Install Docker:
```
# Install the official docker-ce package
sudo curl -sSL https://get.docker.com/ | sh
# Start Docker
sudo systemctl start docker
```
Nvidia Docker is a necessary plugin that enables Nvidia GPU-accelerated applications to run in Docker.

Install nvidia-docker for your Linux platform by following the [installation instructions](https://github.com/NVIDIA/nvidia-docker) of the nvidia-docker repository. On CentOS/RHEL, you should follow section "CentOS 7 (**docker-ce**), RHEL 7.4/7.5 (**docker-ce**), Amazon Linux 1/2" of the repository.

Build the docker image from the [Dockerfile](/Plugins/Server/Dockerfile):
```
# Start by loading Ubuntu16.04 with cuda 9.0 and cudnn7 as the base image
sudo docker pull nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Build the docker image on top of the base image
cd Plugins/Server/
# Choose your own label for <docker_image_name>, it must be lowercase. e.g. mlserver.
sudo docker build -t <docker_image_name> -f Dockerfile .
```

### Run Docker Container

Create and run a docker container on top of the created docker image, referencing the `<docker_image_name>` from the previous step:

```
sudo nvidia-docker run -v /absolute/path/to/nuke-ML-server/Models/:/workspace/ml-server/models:ro -it <docker_image_name>
```

Note: the `-v` (volume) option links your host machine Models/ folder with the models/ folder inside your container. You only need to modify `/absolute/path/to/nuke-ML-server/Models/`, leave the `/workspace/ml-server/models:ro` unchanged as it already corresponds to the folder structure inside your Docker image. This option allows you to add models in Models/ that will be directly available and updated inside your container.

If you get the following error:
```
/bin/nvidia-docker: line 34: /bin/docker: Permission denied
/bin/nvidia-docker: line 34: /bin/docker: Success
```
Try to replace the previous command with:
```
sudo docker run --runtime=nvidia -v /absolute/path/to/nuke-ML-server/Models/:/workspace/ml-server/models:ro -it <docker_image_name>
```

## Getting Started

### Download Configuration and Weights Files

To be able to run inference on the Mask-RCNN model, you need to download its configuration and weight files.

Depending on your GPU memory, you can use either a ResNet101 (GPU memory > 8GB) or a ResNet50 (GPU memory > 6GB) backbone. The results with ResNet101 are slightly better.
- Mask-RCNN requires ~7GB GPU RAM with ResNet101 and ~4.6GB with ResNet50.

Download your selected configuration and weight files:
- Mask-RCNN ResNet50:
  - Configuration: [e2e_mask_rcnn_R-50-FPN_2x.yaml](https://raw.githubusercontent.com/facebookresearch/Detectron/master/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml)
  - Corresponding weights: [model_final.pkl](https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) (from the Detectron [Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md))
- OR Mask_RCNN ResNet101
  - Configuration: [e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml](https://raw.githubusercontent.com/facebookresearch/Detectron/master/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml)
  - Correponding weights: [model_final.pkl](https://dl.fbaipublicfiles.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) (from the Detectron [Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md))

And move them to `Models/mrcnn/` folder.

ResNet50 is the default backbone. If you use ResNet101, you need to modify the config and weight file names in Models/mrcnn/model.py.

### Connect Client and Server

Finally to connect the Python server with the Nuke client:

0. (If you have stopped your container, follow the [Run Docker Container](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/INSTALL.md#run-docker-container) section again)
1. In the running docker container, query the ip address:
```
hostname -I
```
2. In Nuke, set the MLClient node host to the container ip address,
3. In the container, launch the server and start listening on port 55555:
```
python server.py 55555
```
4. In Nuke, click on the MLClient connect button, you should have the three models available.

### Add your own Model

To implement your own model, you can create a new folder in the /Models directory with your model name. At the minimum, this folder needs to include an empty `__init__.py` file and a `model.py` file that contains a Model class inheriting from BaseModel.

You can copy the simple [Models/blur/](Models/blur) model as a starting point, and implement your own model looking at the examples of blur and mrcnn.
