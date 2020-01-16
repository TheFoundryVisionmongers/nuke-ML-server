# Regression Training Template

The regressionTemplateTF is a training template written in TensorFlow. It aims at quickly enabling image-to-parameters training. For instance, finding the lens distortion parameters or gamma correction of an image. When trained, the model can be tested and used directly in Nuke through the nuke-ML-server.

Compared to the image-to-image [Training Template](https://github.com/TheFoundryVisionmongers/nuke-ML-server/tree/master/Models/trainingTemplateTF) and the image-to-labels [Classification Template](https://github.com/TheFoundryVisionmongers/nuke-ML-server/tree/master/Models/classTemplateTF), this template will not work out-of-the-box and will require some data preprocessing implementation, as detailed in the [following section](https://github.com/TheFoundryVisionmongers/nuke-ML-server/tree/master/Models/regressionTemplateTF#data-preprocessing-implementation). This guide will be based on the current template example: gamma-correction prediction.

For instructions on how to set-up the training, on potential training issues or on TensorBoard visualisation, please refer to the [training template readme](https://github.com/TheFoundryVisionmongers/nuke-ML-server/blob/master/Models/trainingTemplateTF/README.md).

## Data Preprocessing Implementation

To train the ML algorithm, you need to set-up your dataset in `regressionTemplateTF/data/train/`. In addition to the training data, it is highly recommended to have validation data in `regressionTemplateTF/data/validation/`. This allows you to check that there is no overfitting on the training data. Please note that the validation dataset and training dataset must not intersect.

Your training/validation dataset will be different depending on your task, i.e. depending on which parameter(s) you want to learn. In the current implementation, we are doing a regression on one parameter (gamma) with a specifically designed data preprocessing pipeline. Namely our model training input is a stack of both original and gamma-graded image histograms.

Our preprocessing pipeline read the original image (from `regressionTemplateTF/data/train/` or `regressionTemplateTF/data/validation/`), then apply gamma correction to that image using a random gamma value. Both the original and resulting gamma-graded images are grayscaled, resized and we compute their 100-bin histogram. The model input (shape [2, 100]) is a stack of those two histograms.

The above data preprocessing is specific to the gamma-correction problem, which means that for other parameters prediction (e.g. colour grading, lens distortion..), you will have to modifiy the data preprocessing functions found in `train_regression.py` and in `model.py` to match your task. The inference file `model.py` has to be changed as well, as the same data preprocessing used in training has to be applied before doing an inference in Nuke.

To summarise, for your specific regression task, you need to implement an appropriate data preprocessing and modify the code in both the training file `train_regression.py` and the inference file `model.py` accordingly.

## Training

Inside your docker container, go to the regressionTemplateTF folder:
```
cd /workspace/ml-server/models/regressionTemplateTF
```
Then directly train your model:
```
python train_regression.py
```
You can also specify the batch size, learning rate and number of epochs:
```
python train_regression.py --bch=16 --lr=1e-3 --ep=1000
```
It is now possible to have deterministic training. You will be able to reproduce your training (get same model weights) by setting the seed to a random int number (here 77):
```
python train_regression.py --seed=77
```
We enable deterministic training in part by applying a GPU patch to the stock TensorFlow, this GPU patch slows down training significantly. By adding the `--no-gpu-patch` tag to the previous command, you achieve a slighlty less deterministic training but keep the same training time.

Note: the current gamma-correction task is creating gamma-graded images on-the-fly using random gamma values, so for the training to succeed it is recommended to have >500 training images.




