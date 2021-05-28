# Installing Nuke Machine Learning Plugin

The Nuke Machine Learning (ML) installation can be divided into compiling the MLClient Nuke node and installing the MLServer using Docker.

The MLClient plugin can be compiled on both Linux/MacOS and Windows systems. It communicates with the MLServer which needs to be run on a Linux machine with NVIDIA GPU.

**Requirements:**
- Linux with Nuke installed
- NVIDIA GPU (Important: GPU memory must be at least 6GB)
- CMake (minimum 3.10)
- Protobuf (tested with 2.5.0 and 3.5.1)
- Docker

## Installing the Client on Linux/MacOS

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
Execute the commands below to compile the client MLClient.so plugin, setting the NUKE_INSTALL_PATH to point to the folder of the desired Nuke version:
```
cd nuke-ML-server/
mkdir build && cd build
cmake -DNUKE_INSTALL_PATH=/path/to/Nuke11.3v1/ ..
make
```
The MLClient.so plugin will now be in the `build/Plugins/Client` folder. Before it can be used, Nuke needs to know where it lives. One way to do this is to update the NUKE_PATH environment variable to point to the MLClient.so plugin (This can be skipped if it was moved to the root of your ~/.nuke folder, or the path was added in Nuke through Python):
```
export NUKE_PATH=/path/to/lib/:$NUKE_PATH
```
At that point, after opening Nuke and updating all plugins, the `MLClient` node should be available. To update all the plugins in Nuke, you can either use the Other > All Plugins > Update option (see [documentation](https://learn.foundry.com/nuke/developers/63/pythondevguide/installing_plugins.html)), or simply press `tab` in the Node Graph then write `Update [All plugins]`. If the `MLClient` node is still missing, verify that the current NUKE_PATH is correctly pointing to the folder containing MLClient.so.

## Installing the Client on Windows

This was tested on Windows 10. You need to have [cmake](https://cmake.org/) and [git](https://git-scm.com/) installed on your computer.

Start by installing the Visual Studio Compiler "Build Tools for Visual Studio 2017" found at [this link](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15).

### Install Protobuf

We recommend building Protobuf locally as a static library. For reference this section partly follows the [installation instructions](https://github.com/protocolbuffers/protobuf/blob/master/cmake/README.md) from the Protobuf GitHub repository.

First open “**x64** Native Tools Command Prompt for VS 2017” executable. Please note it has to be **x64** and not x86.

If `cmake` or `git` commands are not available from Command Prompt, add them to the system PATH variable:
```
set PATH=%PATH%;C:\Program Files (x86)\CMake\bin
set PATH=%PATH%;C:\Program Files\Git\cmd
```
Clone your chosen Protobuf branch release, for instance here version 3.5.1:
```
git clone -b v3.5.1 https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
cd cmake
mkdir build & cd build
mkdir release & cd release
```
Compile protobuf with dynamic VCRTLib (Visual Studio Code C++ Runtime Library):
```
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<protobuf_install_dir> -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF ../..
```
Install protobuf in the specified `<protobuf_install_dir>` folder by running the following:
```
nmake install
```
Note: This last command will create the following folders under the `<protobuf_install_dir>` location:
- bin - that contains protobuf protoc.exe compiler;
- include - that contains C++ headers and protobuf *.proto files;
- lib - that contains linking libraries and CMake configuration files for protobuf package.

### Compile MLClient Nuke Node

If not already done, clone the `nuke-ML-server` repository:
```
git clone https://github.com/TheFoundryVisionmongers/nuke-ML-server
cd nuke-ml-server
mkdir build & cd build
mkdir x64-Release & cd x64-Release
```
Compile the MLClient and link your version of Nuke and Protobuf install path:
```
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DNUKE_INSTALL_PATH=”/path/to//Nuke12.0v3” -DProtobuf_LIBRARIES=”<protobuf_install_dir>/lib” -DProtobuf_INCLUDE_DIR=”<protobuf_install_dir>/include” -DProtobuf_PROTOC_EXECUTABLE="<protobuf_install_dir>/bin/protoc.exe" ../..
nmake
```
The MLClient.dll plugin should now be in the `build/x64-Release/Plugins/Client` folder. Before it can be used, Nuke needs to know where it lives. You can either copy it to your ~/.nuke folder or update the NUKE_PATH environment:
```
set NUKE_PATH=%NUKE_PATH%;path/to/lib
```
At that point, after opening Nuke and updating all plugins, the `MLClient` node should be available. To update all the plugins in Nuke, you can either use the Other > All Plugins > Update option (see [documentation](https://learn.foundry.com/nuke/developers/63/pythondevguide/installing_plugins.html)), or simply press `tab` in the Node Graph then write `Update [All plugins]`. If the `MLClient` node is still missing, verify that the current NUKE_PATH is correctly pointing to the folder containing MLClient.dll.

As your client is on a Windows machine, you now need to run the server on a Linux machine with NVidia GPU (see [next section](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/INSTALL.md#installing-the-server)) and connect your Windows machine to it following the [Connect to an External Server](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/INSTALL.md#connect-to-an-external-server) section.

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

Install nvidia-container-toolkit for your Linux platform by following the [installation instructions](https://github.com/NVIDIA/nvidia-docker) of the nvidia-docker repository. On CentOS/RHEL, you should follow section "CentOS 7 (**docker-ce**), RHEL 7.4/7.5 (**docker-ce**), Amazon Linux 1/2" of the repository.

Build the docker image from the [Dockerfile](/Plugins/Server/Dockerfile):
```
# Start by loading Ubuntu18.04 with cuda 10.0 and cudnn7 as the base image
sudo docker pull nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# Build the docker image on top of the base image
cd Plugins/Server/
# Choose your own label for <docker_image_name>, it must be lowercase. e.g. mlserver.
sudo docker build -t <docker_image_name> -f Dockerfile .
```

### Run Docker Container

Create and run a docker container on top of the created docker image, referencing the `<docker_image_name>` from the previous step:

```
sudo docker run --gpus all -v /absolute/path/to/nuke-ML-server/Models/:/workspace/ml-server/models -it <docker_image_name>
```

Notes:
- the `-v` (volume) option links your host machine Models/ folder with the models/ folder inside your container. You only need to modify `/absolute/path/to/nuke-ML-server/Models/`, leave the `/workspace/ml-server/models` unchanged as it already corresponds to the folder structure inside your Docker image. This option allows you to add models in Models/ that will be directly available and updated inside your container.
- If your docker version doesn't recognise the `--gpus` flag, you can equally run the same docker container by replacing `sudo docker run --gpus all ` by `sudo nvidia-docker run` or `sudo docker run --runtime=nvidia`.

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

This section explains how to connect the server and client when your docker container and Nuke instance are running on the same Linux machine:

0. (If you have stopped your container, follow the [Run Docker Container](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/INSTALL.md#run-docker-container) section again)
1. In the running docker container, query the IP address:
```
hostname -I
```
2. In Nuke, set the MLClient node `host` to the container IP address,
3. In the container, launch the server and start listening on port 55555:
```
python server.py 55555
```
4. In Nuke, click on the MLClient connect button, you should have the three models available.

### Connect to an External Server

This section explains how to connect server and client when your docker container (MLServer) and Nuke (MLClient) are running on two different machines, e.g. if you are using the MLClient on Windows. In that case, you have a Linux machine running the docker container and a Windows machine running Nuke.

1. On your **Linux machine** (not the docker container, not your Windows machine), query the IP adress:
```
hostname -I
```
2. In Nuke, set the MLClient node `host` to the Linux machine IP address obtained.
3. On the Linux machine, run the docker container exporting a port of your choice (here port 7000 of the host is mapped to port 55555 of the container):
```
sudo docker run --gpus all -v /absolute/path/to/nuke-ML-server/Models/:/workspace/ml-server/models -p 7000:55555 -it <docker_image_name>
```
4. In the container, launch the server and start listening on port 55555:
```
python server.py 55555
```
5. In Nuke, set the MLClient node `port` to 7000 and click on the MLClient connect button.

### Add your own Model

To implement your own model, you can create a new folder in the /Models directory with your model name. At the minimum, this folder needs to include an empty `__init__.py` file and a `model.py` file that contains a Model class inheriting from BaseModel.

You can copy the simple [Models/blur/](Models/blur) model as a starting point, and implement your own model looking at the examples of blur and mrcnn.
