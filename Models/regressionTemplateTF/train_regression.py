# Copyright (c) 2020 Foundry.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import division, print_function, absolute_import
from builtins import input # python 2/3 forward-compatible (raw_input)

import sys
import os
import time
import random
import argparse
from datetime import datetime

import numpy as np

import tensorflow as tf
print(tf.__version__)

tf.compat.v1.enable_eager_execution()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_builder import baseline_model
from common.util import get_filepaths_from_dir, get_ckpt_list, print_
from common.util import is_exr, read_resize_exr, linear_to_srgb

def enable_deterministic_training(seed, no_gpu_patch=False):
    """Set all seeds for deterministic training

    Args:
        no_gpu_patch (bool): if False, apply a patch to TensorFlow to have
            deterministic GPU operations, if True the training is much faster
            but slightly less deterministic.
    This function needs to be called before any TensorFlow code.
    """
    import numpy as np
    import os
    import random
    import tfdeterminism
    if not no_gpu_patch:
        # Patch stock TensorFlow to have deterministic GPU operation
        tfdeterminism.patch() # then use tf as normal
    # If PYTHONHASHSEED environment variable is not set or set to random,
    # a random value is used to seed the hashes of str, bytes and datetime
    # objects. (Necessary for Python >= 3.2.3)
    os.environ['PYTHONHASHSEED']=str(seed)
    # Set python built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # Set seed for random Numpy operation (e.g. np.random.randint)
    np.random.seed(seed)
    # Set seed for random TensorFlow operation (e.g. tf.image.random_crop)
    tf.compat.v1.random.set_random_seed(seed)

## DATA PROCESSING

def histogram(tensor, value_range=[0.0, 1.0], nbins=100):
    """Return histogram of tensor"""
    h, w, c = tensor.shape
    hist = tf.histogram_fixed_width(tensor, value_range, nbins=nbins)
    hist = tf.divide(hist, h * w * c)
    return hist

def gamma_correction(img, gamma):
    """Apply gamma correction to image img
    
    Returns:
        hists: stack of both original and graded image histograms
    """
    # Check number of parameter is one
    if gamma.shape[0] != 1:
        raise ValueError("Parameter for gamma correction must be of "
            "size (1,), not {}.\n\tCheck your self.output_param_number, ".format(gamma.shape)
            + "you may need to implement your own input_data preprocessing.")
    # Create groundtruth graded image
    img_grade = tf.math.pow(img, gamma)
    # Compute histograms
    img_hist = histogram(img)
    img_grade_hist = histogram(img_grade)
    hists = tf.stack([img_hist, img_grade_hist], axis=0)
    return hists

## CUSTOM TRAINING METRICS

def bin_acc(y_true, y_pred, delta=0.02):
    """Bin accuracy metric equals 1.0 if diff between true
    and predicted value is inferior to delta.
    """
    diff = tf.keras.backend.abs(y_true - y_pred)
    # If diff is less that delta --> true (1.0), otherwise false (0.0)
    correct = tf.keras.backend.less(diff, delta)
    # Return percentage accuracy
    return tf.keras.backend.mean(correct)

class TrainModel(object):
    """Train Regression model from the given data"""

    def __init__(self, args):
        # Training hyperparameters
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.patch_size = 50
        self.channels = 3 # input / output channels
        self.output_param_number = 1
        self.no_resume = args.no_resume
        # A random seed (!=None) allows you to reproduce your training results
        self.seed = args.seed
        if self.seed is not None:
            # Set all seeds necessary for deterministic training
            enable_deterministic_training(self.seed, args.no_gpu_patch)
        # Training and validation dataset paths
        train_data_path = './data/train/'
        val_data_path = './data/validation/'

        # Where to save and load model weights (=checkpoints)
        self.ckpt_dir = './checkpoints'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.ckpt_save_name = args.ckpt_save_name

        # Where to save tensorboard summaries
        self.summaries_dir = './summaries/'
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)

        # Get training dataset as list of image paths
        self.train_data_list = get_filepaths_from_dir(train_data_path)
        if not self.train_data_list:
            raise ValueError("No training data found in folder {}".format(train_data_path))
        elif (len(self.train_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of training data = {})"
                .format(self.batch_size, len(self.train_data_list)))
        self.is_exr = is_exr(self.train_data_list[0])

        # Compute and print training hyperparameters
        self.batch_per_epoch = (len(self.train_data_list)) // self.batch_size
        max_steps = int(self.epoch * (self.batch_per_epoch))
        print_("Number of training data: {}\nNumber of batches per epoch: {} (batch size = {})\nNumber of training steps for {} epochs: {}\n"
            .format(len(self.train_data_list), self.batch_per_epoch, self.batch_size, self.epoch, max_steps), 'm')

        # Get validation dataset if provided
        self.has_val_data = True
        self.val_data_list = get_filepaths_from_dir(val_data_path)
        if not self.val_data_list:
            print("No validation data found in {}".format(val_data_path))
            self.has_val_data = False
        elif (len(self.val_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of validation data = {})"
                .format(self.batch_size, len(self.val_data_list)))
        else:
            val_is_exr = is_exr(self.val_data_list[0])
            if (val_is_exr and not self.is_exr) or (not val_is_exr and self.is_exr):
                raise TypeError("Train and validation data should have the same file format")
            self.val_batch_per_epoch = (len(self.val_data_list)) // self.batch_size
            print("Number of validation data: {}\nNumber of validation batches per epoch: {} (batch size = {})"
                .format(len(self.val_data_list), self.val_batch_per_epoch, self.batch_size))

    def get_data(self, data_list, batch_size=16, epoch=100, shuffle_buffer_size=1000):
        
        def read_and_preprocess_data(path_img, param):
            """Read image in path_img, resize it to patch_size,
            convert to grayscale and apply a random gamma grade to it

            Returns:
                input_data: stack of both original and graded image histograms
                param: groundtruth gamma value
            """
            if self.is_exr: # ['exr', 'EXR']
                img = tf.numpy_function(read_resize_exr,
                    [path_img, self.patch_size], [tf.float32])
                img = tf.numpy_function(linear_to_srgb, [img], [tf.float32])
                img = tf.reshape(img, [self.patch_size, self.patch_size, self.channels])
                img = tf.image.rgb_to_grayscale(img)
            else: # ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']
                img_raw = tf.io.read_file(path_img)
                img_tensor = tf.image.decode_png(img_raw, channels=3)
                img = tf.cast(img_tensor, tf.float32) / 255.0
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.resize(img, [self.patch_size, self.patch_size])
            # Depending on what parameter(s) you want to learn, modify the training
            # input data. Here to learn gamma correction, our input data trainX is
            # a stack of both original and gamma-graded histograms.
            input_data = gamma_correction(img, param)
            return input_data, param

        with tf.compat.v1.variable_scope('input'):
            # Ensure preprocessing is done on the CPU (to let the GPU focus on training)
            with tf.device('/cpu:0'):
                data_tensor = tf.convert_to_tensor(data_list, dtype=tf.string)
                path_dataset = tf.data.Dataset.from_tensor_slices((data_tensor))
                path_dataset = path_dataset.shuffle(shuffle_buffer_size).repeat(epoch)
                # Depending on what parameter(s) you want to learn, modify the random
                # uniform range. Here create random gamma values between 0.2 and 5
                param_tensor = tf.random.uniform(
                    [len(data_list)*epoch, self.output_param_number], 0.2, 5.0)
                param_dataset = tf.data.Dataset.from_tensor_slices((param_tensor))
                dataset = tf.data.Dataset.zip((path_dataset, param_dataset))
                # Apply read_and_preprocess_data function to all input in the path_dataset
                dataset = dataset.map(read_and_preprocess_data, num_parallel_calls=4)
                dataset = dataset.batch(batch_size)
                # Always prefetch one batch and make sure there is always one ready
                dataset = dataset.prefetch(buffer_size=1)
                return dataset

    def tensorboard_callback(self, writer):
        """Return custom Tensorboard callback for logging main metrics"""

        def log_metrics(epoch, logs):
            """Log training/validation loss and accuracy to Tensorboard"""
            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('train_loss', logs['loss'], step=epoch)
                tf.contrib.summary.scalar('train_bin_acc', logs['bin_acc'], step=epoch)
                if self.has_val_data:
                    tf.contrib.summary.scalar('val_loss', logs['val_loss'], step=epoch)
                    tf.contrib.summary.scalar('val_bin_acc', logs['val_bin_acc'], step=epoch)
                tf.contrib.summary.flush()

        return tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metrics)

    def get_compiled_model(self, input_shape):
        model = baseline_model(
            input_shape,
            output_param_number=self.output_param_number)
        adam = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=adam,
            loss='mean_squared_error',
            metrics=[bin_acc])
        return model

    def train(self):
        # Create a session so that tf.keras don't allocate all GPU memory at once
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        tf.compat.v1.keras.backend.set_session(sess)
        
        # Get training and validation dataset
        ds_train = self.get_data(
            self.train_data_list,
            self.batch_size,
            self.epoch)
        for x, y in ds_train.take(1): # take one batch from ds_train
            trainX, trainY = x, y
        print("Input shape {}, target shape: {}".format(trainX.shape, trainY.shape))
        if self.has_val_data:
            ds_val = self.get_data(
                self.val_data_list,
                self.batch_size,
                self.epoch)   
        print("********Data Created********")

        # Build model
        model = self.get_compiled_model(trainX.shape[1:])

        # Check if there are intermediate trained model to load
        if self.no_resume or not self.load(model):
            print_("Starting training from scratch\n", 'm')

        # Callback for creating Tensorboard summary
        summary_name = ("data{}_bch{}_ep{}".format(
            len(self.train_data_list), self.batch_size, self.epoch))
        summary_name += ("_seed{}".format(self.seed) if self.seed is not None else "")
        summary_writer = tf.contrib.summary.create_file_writer(
            os.path.join(self.summaries_dir, summary_name))
        tb_callback = self.tensorboard_callback(summary_writer)

        # Callback for saving model's weights
        ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_save_name + "-ep{epoch:02d}")
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            # save best model based on monitor value
            monitor='val_loss' if self.has_val_data else 'loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        
        # Evaluate the model before training
        if self.has_val_data:
            val_loss, val_bin_acc = model.evaluate(ds_val.take(20), verbose=1)
            print("Initial Loss on validation dataset: {:.4f}".format(val_loss))

        # TRAIN model
        print_("--------Start of training--------\n", 'm')
        print("NOTE:\tDuring training, the latest model is saved only if its\n"
            "\t(validation) loss is better than the last best model.")
        train_start = time.time()
        model.fit(
            ds_train,
            validation_data=ds_val if self.has_val_data else None,
            epochs=self.epoch,
            steps_per_epoch=self.batch_per_epoch,
            validation_steps=self.val_batch_per_epoch if self.has_val_data else None,
            callbacks=[ckpt_callback, tb_callback],
            verbose=1)
        print_("Training duration: {:0.4f}s\n".format(time.time() - train_start), 'm')
        print_("--------End of training--------\n", 'm')

        # Show predictions on the first batch of training data
        print("Parameter prediction (PR) compared to groundtruth (GT) for first batch of training data:")
        preds_train = model.predict(trainX.numpy())
        print("Train GT:", trainY.numpy().flatten())
        print("Train PR:", preds_train.flatten())
        # Make predictions on the first batch of validation data
        if self.has_val_data:
            print("For first batch of validation data:")
            for x, y in ds_val.take(1): # take one batch from ds_val
                valX, valY = x, y
            preds_val = model.predict(valX)
            print("Val GT:", valY.numpy().flatten())
            print("Val PR:", preds_val.flatten())
        # Free all resources associated with the session
        sess.close()

    def load(self, model):
        ckpt_names = get_ckpt_list(self.ckpt_dir)
        if not ckpt_names: # list is empty
            print_("No checkpoints found in {}\n".format(self.ckpt_dir), 'm')
            return False
        else:
            print_("Found checkpoints:\n", 'm')
            for name in ckpt_names:
                print("    {}".format(name))
            # Ask user if they prefer to start training from scratch or resume training on a specific ckeckpoint 
            while True:
                mode=str(input('Start training from scratch (start) or resume training from a previous checkpoint (choose one of the above): '))
                if mode == 'start' or mode in ckpt_names:
                    break
                else:
                    print("Answer should be 'start' or one of the following checkpoints: {}".format(ckpt_names))
                    continue
            if mode == 'start':
                return False
            elif mode in ckpt_names:
                # Try to load given intermediate checkpoint
                print_("Loading trained model...\n", 'm')
                model.load_weights(os.path.join(self.ckpt_dir, mode))
                print_("...Checkpoint {} loaded\n".format(mode), 'm')
                return True
            else:
                raise ValueError("User input is neither 'start' nor a valid checkpoint")

    def evaluate(self, test_data_path, weights):
        """Evaluate a trained model on the test dataset
        
        Args:
            test_data_path (str): path to directory containing images for testing
            weights (str): name of the tensorflow checkpoint (weights) to evaluate
        """
        test_data_list = get_filepaths_from_dir(test_data_path)
        if not test_data_list:
            raise ValueError("No test data found in folder {}".format(test_data_path))
        elif (len(self.train_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of test data = {})"
                .format(self.batch_size, len(test_data_list)))
        self.is_exr = is_exr(test_data_list[0])

        # Get and create test dataset
        ds_test = self.get_data(
            test_data_list,
            self.batch_size,
            1)
        for x, y in ds_test.take(1): # take one batch from ds_test
            testX, testY = x, y
        print_("Number of test data: {}\n".format(len(test_data_list)), 'm')
        print("Input shape {}, target shape: {}".format(testX.shape, testY.shape))

        # Build model
        model = self.get_compiled_model(testX.shape[1:])

        # Load model weights
        print_("Loading trained model for testing...\n", 'm')
        model.load_weights(os.path.join(self.ckpt_dir, weights)).expect_partial()
        print_("...Checkpoint {} loaded\n".format(weights), 'm')

        # Test final model on this unseen dataset
        results = model.evaluate(ds_test)
        print("test loss, test acc:", results)
        print_("--------End of testing--------\n", 'm')

def parse_args():
    parser = argparse.ArgumentParser(description='Model training arguments')
    parser.add_argument('--bch', type=int, default=10, dest='batch_size', help='training batch size')
    parser.add_argument('--ep', type=int, default=15, dest='epoch', help='training epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--seed', type=int, default=None, dest='seed', help='set random seed for deterministic training')
    parser.add_argument('--no-gpu-patch', dest='no_gpu_patch', default=False, action='store_true', help='if seed is set, add this tag for much faster but slightly less deterministic training')
    parser.add_argument('--no-resume', dest='no_resume', default=False, action='store_true',  help="start training from scratch")
    parser.add_argument('--name', type=str, default="regressionTemplateTF", dest='ckpt_save_name', help='name of saved checkpoints/model weights')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # Set up model to train
    model = TrainModel(args)
    model.train()
    # To evaluate on the test dataset, uncomment next line and give the
    # test dataset directory and the model checkpoint name
    # model.evaluate('./data/test', 'regressionTemplateTF-ep35')