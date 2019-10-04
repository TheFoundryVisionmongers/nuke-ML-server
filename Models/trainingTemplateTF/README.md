# Training Template: Train and Infer Models in the nuke-ML-server

The TrainingTemplateTF model is a training template written in TensorFlow. It aims at quickly enabling image-to-image training using a multi-scale encoder-decoder model. When trained, the model can be tested and used directly in Nuke through the nuke-ML-server.

For instance, if you have a set of noisy / clear image pairs and would like to train a model to be able to denoise an image, you simply need to fill in your data in the `TrainingTemplateTF/data` and start the training with one command line. You can monitor the training using TensorBoard and eventually test the trained model on live images in Nuke.

This page contains instructions on how to use this training template. The training happens in the Docker container, while the inference is done through the MLClient plugin.

## Set-up

Start by installing the nuke-ML-server (see [INSTALL.md](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/INSTALL.md)). If you had already installed the previous version, you will still have to rebuild the docker image once:
```
cd Plugins/Server/
sudo docker build -t <docker_image_name> -f Dockerfile .
```

To launch the [TensorBoard Visualisation](https://github.com/TheFoundryVisionmongers/nuke-ML-server/tree/master/Models/trainingTemplateTF#tensorboard) from within the Docker, you have to run the docker container ([Run Docker Container](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/INSTALL.md#run-docker-container) section) with an exported port 6006:
```
sudo docker run --gpus all -v /absolute/path/to/nuke-ML-server/Models/:/workspace/ml-server/models -p 6006:6006 -it <docker_image_name>
```

## Train in Docker

### Dataset

To train the ML algorithm, your need to provide a dataset of groundtruth & input image data pairs. For instance, the input data could be blurred images and the groundtruth corresponding sharp images. In that case, you would like the model to learn to infer a sharp image out of a blurred input image.

Respectively place your input and groundtruth data in `trainingTemplateTF/data/train/input/` and `trainingTemplateTF/data/train/groundtruth/` folders.

Optionally, you can add a separate set of image pairs in `trainingTemplateTF/data/val/input/`and `trainingTemplateTF/data/val/groundtruth/`. If this validation dataset is available, it is periodically tested on the current model weights to check that there is no overfitting on the training data. Please note that the validation dataset and training dataset must not intersect, no image pair should be found in both datasets.

Notes:
- The preprocessing cropping size is currently 256x256, therefore the dataset images are expected to be at least 256x256.
- Supported image types are JPG, PNG, BMP and EXR.
- Depending on the compression used, EXR images can be slower to read. In our experiments, the fastest EXR read is achieved with B44, B44A or no compression.

### Training

Inside your docker container, go to the trainingTemplateTF folder:
```
cd /workspace/ml-server/models/trainingTemplateTF
```
Then directly train your model:
```
python train_model.py
```
You can also specify the batch size, learning rate and number of epochs:
```
python train_model.py --bch=16 --lr=1e-4 --ep=10000
```
### Potential Training Issues

The principal issue you may hit when training is a GPU out-of-memory (OOM) error. To apply training with default values, your GPU memory should be at least 8GB.

If you reach an OOM error, you can consider reducing the GPU memory requirements -likely at the expense of the final model performance- by:
- Building a simplified version of the encoder-decoder model found in [`model_builder.py`](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/Models/trainingTemplateTF/util/model_builder.py) (e.g. by removing layers),
- Reducing the batch size (`--bch` argument),
- Or lowering the preprocessing cropping size (`crop_size` in [`train_model.py`](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/Models/trainingTemplateTF/train_model.py)).

During training, images are cropped as a preprocessing step before being fed to the network. Therefore if you want your model to learn a global image information (e.g. lens distortion), this cropping preprocessing should be changed in the code (e.g. use resize & padding instead), so as to keep the whole image information.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) is a great way to visualise how your training is progressing.

The TrainingTemplateTF automatically saves learning rate and loss evolution as well as input, groundtruth and temporary output images in the `trainingTemplateTF/summaries/` folder.

To view these TensorBoard summaries, first find which container is currently running your training (STATUS: Up, PORTS: 0.0.0.0:6006->6006/tcp, NAMES=`<container>`) from all the created docker containers:
```
sudo docker ps -a
```
Launch a second terminal connected to the same docker container, where `<container>` is the name of your training container found above:
```
docker exec -it <container> bash
```
Launch TensorBoard in this new docker terminal to view the progression in real-time in your browser:
```
tensorboard --logdir models/trainingTemplateTF/summaries/
```
From your host machine, you can now navigate to the following browser address to monitor your training: http://localhost:6006.

### Checkpoints

During training, the model weights and graph are saved every N steps and put in the `trainingTemplateTF/checkpoints/` folder. A checkpoint name, for instance `trainingTemplateTF.model-375000` means that it contains the weights after 375,000 training steps using model trainingTemplateTF.

When launching a training, you can decide to start from scratch or resume training from a list of previous checkpoints.

## Inference in Nuke

After training your model inside the docker container, you can launch Nuke and select the `Training Template TF` model in the MLClient node.

The plugin will automatically load the most advanced trained checkpoints found in `trainingTemplateTF/checkpoints/`, and run an inference using the loaded weights and graph. If you prefer to use older checkpoints, you can write the name of a previous checkpoint as an inference option in Nuke. 

This is a great way to verify on your own live-data that the model weights converged correctly without overfitting on the training data.

Note: the inference is done on saved checkpoints and not on a frozen graph, which implies that the saved checkpoint graph must correspond to the current graph. If you change the graph (by changing the preprocessing step, number of layers, variable names etc.), you won't directly be able to load older checkpoints built on a different graph.