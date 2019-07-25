# Classification Training Template

The classTemplateTF is a training template written in TensorFlow. It aims at quickly enabling classification training. For instance, detecting the presence of a specific actor in a shot. When trained, the model can be tested and used directly in Nuke through the nuke-ML-server.

Apart from the dataset structure, all other instructions are similar to the [Training Template](https://github.com/TheFoundryVisionmongers/nuke-ML-server/tree/master/Models/trainingTemplateTF).

## Dataset

To train the ML algorithm, you need to set-up your dataset in `classTemplateTF/data/train/`. This directory should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories will be included.

For example, if you want to train a classifier to differentiate between cats, dogs and foxes. The `data/train/` directory should have 3 subdirectories named `cats`, `dogs` and `foxes` with each directory containing images of the corresponding animal.

Optionally, you can add a separate set of images in `classTemplateTF/data/validation/`. If available, it is periodically used to check that there is no overfitting on the training data. Please note that the validation dataset and training dataset must not intersect.

If no validation dataset is found, 20% of the training data will be used as a validation split.

