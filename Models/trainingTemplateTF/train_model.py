# Copyright (c) 2019 Foundry.
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

from __future__ import print_function

import sys
import os
import time
import random
import argparse
from datetime import datetime

import numpy as np

import tensorflow as tf
print(tf.__version__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_builder import EncoderDecoder
from common.util import im2uint8, get_filepaths_from_dir, get_ckpt_list, print_
from common.util import is_exr, read_crop_exr_pair, linear_to_srgb

def enable_deterministic_training(seed, no_gpu_patch=False):
    """Set all seeds for deterministic training
    This function needs to be called before any tensorflow code.
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

class TrainModel(object):
    """Train the EncoderDecoder from the given input and groundtruth data"""

    def __init__(self, args):
        # Training hyperparameters
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.no_resume = args.no_resume
        # A random seed (!=None) allows you to reproduce your training results
        self.seed = args.seed
        if self.seed is not None:
            # Set all seeds necessary for deterministic training
            enable_deterministic_training(self.seed, args.no_gpu_patch)
        self.crop_size = 128
        self.n_levels = 3
        self.scale = 0.5
        self.channels = 3   # input / output channels
        # Training and validation dataset paths
        train_in_data_path = './data/train/input'
        train_gt_data_path = './data/train/groundtruth'
        val_in_data_path = './data/validation/input'
        val_gt_data_path = './data/validation/groundtruth'

        # Where to save and load model weights (=checkpoints)
        self.checkpoints_dir = './checkpoints'
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.ckpt_save_name = args.ckpt_save_name
        # Maximum number of recent checkpoint files to keep
        self.max_ckpts_to_keep = 50
        # In addition keep one checkpoint file for every N hours of training
        self.keep_ckpt_every_n_hours = 1
        # How often, in training steps. we save model checkpoints
        self.ckpts_save_freq = 1000
        # How often, in training steps. we print training losses to bash
        self.training_print_freq = 10

        # Where to save tensorboard summaries
        self.summaries_dir = './summaries'
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)
        # How often, in training steps. we save tensorboard summaries
        self.summaries_save_freq = 10
        # How often, in secs, we flush the pending tensorboard summaries to disk
        self.summary_flush_secs = 30

        # Get training dataset as lists of image paths
        self.train_in_data_list = get_filepaths_from_dir(train_in_data_path)
        self.train_gt_data_list = get_filepaths_from_dir(train_gt_data_path)
        if not self.train_in_data_list or not self.train_gt_data_list:
            raise ValueError("No training data found in folders {} or {}".format(train_in_data_path, train_gt_data_path))
        elif len(self.train_in_data_list) != len(self.train_gt_data_list):
            raise ValueError("{} ({} data) and {} ({} data) should have the same number of input data"
                .format(train_in_data_path, len(self.train_in_data_list), train_gt_data_path, len(self.train_gt_data_list)))
        elif (len(self.train_in_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of training data = {})"
                .format(self.batch_size, len(self.train_in_data_list)))
        self.is_exr = is_exr(self.train_in_data_list[0])

        # Get validation dataset if provided
        self.has_val_data = True
        self.val_in_data_list = get_filepaths_from_dir(val_in_data_path)
        self.val_gt_data_list = get_filepaths_from_dir(val_gt_data_path)
        if not self.val_in_data_list or not self.val_gt_data_list:
            print("No validation data found in {} or {}".format(val_in_data_path, val_gt_data_path))
            self.has_val_data = False
        elif len(self.val_in_data_list) != len(self.val_gt_data_list):
            raise ValueError("{} ({} data) and {} ({} data) should have the same number of input data"
                .format(val_in_data_path, len(self.val_in_data_list), val_gt_data_path, len(self.val_gt_data_list)))
        elif (len(self.val_in_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of validation data = {})"
                .format(self.batch_size, len(self.val_in_data_list)))
        else:
            val_is_exr = is_exr(self.val_in_data_list[0])
            if (val_is_exr and not self.is_exr) or (not val_is_exr and self.is_exr):
                raise TypeError("Train and validation data should have the same file format")
            print("Number of validation data: {}".format(len(self.val_in_data_list)))
        
        # Compute and print training hyperparameters
        batch_per_epoch = (len(self.train_in_data_list)) // self.batch_size
        self.max_steps = int(self.epoch * (batch_per_epoch))
        print_("Number of training data: {}\nNumber of batches per epoch: {} (batch size = {})\nNumber of training steps for {} epochs: {}\n"
            .format(len(self.train_in_data_list), batch_per_epoch, self.batch_size, self.epoch, self.max_steps), 'm')

    def get_data(self, in_data_list, gt_data_list, batch_size=16, epoch=100):

        def read_and_preprocess(path_img_in, path_img_gt):
            if self.is_exr: # ['exr', 'EXR']
                # Read and crop data
                img_crop = tf.numpy_function(read_crop_exr_pair,
                    [path_img_in, path_img_gt, self.crop_size], [tf.float32, tf.float32])
                img_crop = tf.numpy_function(linear_to_srgb, [img_crop], tf.float32)
                img_crop = tf.unstack(tf.reshape(img_crop, [2, self.crop_size, self.crop_size, self.channels]))
            else: # ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']
                # Read data
                img_in_raw = tf.io.read_file(path_img_in)
                img_gt_raw = tf.io.read_file(path_img_gt)
                img_in_tensor = tf.image.decode_image(img_in_raw, channels=self.channels)
                img_gt_tensor = tf.image.decode_image(img_gt_raw, channels=self.channels)
                # Normalise then crop data
                imgs = [tf.cast(img, tf.float32) / 255.0 for img in [img_in_tensor, img_gt_tensor]]
                img_crop = tf.unstack(tf.image.random_crop(tf.stack(imgs, axis=0),
                    [2, self.crop_size, self.crop_size, self.channels], seed=self.seed), axis=0)
            return img_crop

        def multi_thread_preprocess(path_img_in, path_img_gt):
            """Non-random data preprocessing to be run in a multi-thread map
            Read image in path_img, and normalize it
            """
            if self.is_exr:
                # Do nothing, all preprocessing done in single_thread_preprocess
                return path_img_in, path_img_gt
            else:
                img_in_raw = tf.io.read_file(path_img_in)
                img_gt_raw = tf.io.read_file(path_img_gt)
                img_in_tensor = tf.image.decode_image(img_in_raw, channels=self.channels)
                img_gt_tensor = tf.image.decode_image(img_gt_raw, channels=self.channels)
                # Normalise data
                imgs = [tf.cast(img, tf.float32) / 255.0 for img in [img_in_tensor, img_gt_tensor]]
                return imgs

        def single_thread_preprocess(img_in, img_gt):
            """Random data preprocessing to be run in a one thread map
            Crop image with deterministic TensorFlow (png) or Numpy (exr) seed
            """
            if self.is_exr:
                img_crop = tf.numpy_function(read_crop_exr_pair,
                    [img_in, img_gt, self.crop_size], [tf.float32, tf.float32])
                img_crop = tf.numpy_function(linear_to_srgb, [img_crop], tf.float32)
                img_crop = tf.unstack(tf.reshape(img_crop, [2, self.crop_size, self.crop_size, self.channels]))
            else:
                img_crop = tf.unstack(tf.image.random_crop(tf.stack([img_in, img_gt], axis=0),
                    [2, self.crop_size, self.crop_size, self.channels], seed=self.seed), axis=0)
            return img_crop
        
        with tf.compat.v1.variable_scope('input'):
            # Ensure preprocessing is done on the CPU (to let the GPU focus on training)
            with tf.device('/cpu:0'):
                in_list = tf.convert_to_tensor(in_data_list, dtype=tf.string)
                gt_list = tf.convert_to_tensor(gt_data_list, dtype=tf.string)
        
                path_dataset = tf.data.Dataset.from_tensor_slices((in_list, gt_list))
                path_dataset = path_dataset.shuffle(
                    buffer_size=len(in_data_list), seed=self.seed).repeat(epoch)
                # Apply read_and_preprocess function to all input in the path_dataset
                if self.seed is None:
                    # Run all preprocessing in one dataset.map()
                    num_parallel_calls = 1 if self.is_exr else 4
                    dataset = path_dataset.map(read_and_preprocess, num_parallel_calls)
                else:
                    # Perform the non-random ops in a multi-threaded map()
                    dataset = path_dataset.map(multi_thread_preprocess, num_parallel_calls=4)  
                    # Perform the random ops in a single-threaded map() for
                    # deterministic training when seed is not None
                    dataset = dataset.map(single_thread_preprocess, num_parallel_calls=1)
                dataset = dataset.batch(batch_size)
                # Always prefetch one batch and make sure there is always one ready
                dataset = dataset.prefetch(buffer_size=1)
                # Create operator to iterate over the created dataset
                next_element = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
                return next_element
    
    def loss(self, n_outputs, img_gt):
        """Compute multi-scale loss function"""
        loss_total = 0
        for i in xrange(self.n_levels):
            _, hi, wi, _ = n_outputs[i].shape
            gt_i = tf.image.resize(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean(tf.square(gt_i - n_outputs[i]))
            loss_total += loss
            # Save out images and loss values to tensorboard
            tf.compat.v1.summary.image('out_' + str(i), im2uint8(n_outputs[i]))
        # Save total loss to tensorboard
        tf.compat.v1.summary.scalar('loss_total', loss_total)
        return loss_total

    def validate(self, model):
        total_val_loss = 0.0
        # Get next data from preprocessed validation dataset
        val_img_in, val_img_gt = self.get_data(self.val_in_data_list, self.val_gt_data_list, self.batch_size, -1)
        n_outputs = model(val_img_in, reuse=False)
        val_op = self.loss(n_outputs, val_img_gt)
        # Test results over one epoch
        batch_per_epoch = len(self.val_in_data_list) // self.batch_size
        for batch in xrange(batch_per_epoch):
            total_val_loss += val_op
        return total_val_loss / batch_per_epoch

    def train(self):    
        # Build model
        model = EncoderDecoder(self.n_levels, self.scale, self.channels)

        # Learning rate decay
        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.lr = tf.compat.v1.train.polynomial_decay(
            self.learning_rate, global_step,
            decay_steps=self.max_steps,
            end_learning_rate=0.0,
            power=0.3)
        tf.compat.v1.summary.scalar('learning_rate', self.lr)
        # Training operator
        adam = tf.compat.v1.train.AdamOptimizer(self.lr)

        # Get next data from preprocessed training dataset
        img_in, img_gt = self.get_data(
            self.train_in_data_list,
            self.train_gt_data_list,
            self.batch_size,
            self.epoch)
        print('img_in, img_gt', img_in.shape, img_gt.shape)
        tf.compat.v1.summary.image('img_in', im2uint8(img_in))
        tf.compat.v1.summary.image('img_gt', im2uint8(img_gt))

        # Compute image loss
        n_outputs = model(img_in, reuse=False)
        loss_op = self.loss(n_outputs, img_gt)
        # By default, adam uses the current graph trainable_variables to optimise training,
        # thus train_op should be the last operation of the graph for training.
        train_op = adam.minimize(loss_op, global_step)

        # Create session
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        
        # Initialise all the variables in current session
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        self.saver = tf.compat.v1.train.Saver(
            max_to_keep=self.max_ckpts_to_keep,
            keep_checkpoint_every_n_hours=self.keep_ckpt_every_n_hours)

        # Check if there are intermediate trained model to load
        if self.no_resume or not self.load(sess, self.checkpoints_dir):
            print_("Starting training from scratch\n", 'm')

        # Tensorboard summary
        summary_op = tf.compat.v1.summary.merge_all()
        summary_name = ("data{}_bch{}_ep{}".format(
            len(self.train_in_data_list), self.batch_size, self.epoch))
        summary_name += ("_seed{}".format(self.seed) if self.seed is not None else "")
        summary_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(self.summaries_dir, summary_name),
            graph=sess.graph,
            flush_secs=self.summary_flush_secs)

        # Compute loss on validation dataset to check overfitting
        if self.has_val_data:
            val_loss_op = self.validate(model)
            # Save validation loss to tensorboard
            val_summary_op = tf.compat.v1.summary.scalar('val_loss', val_loss_op)
            # Compute initial loss
            val_loss, val_summary = sess.run([val_loss_op, val_summary_op])
            summary_writer.add_summary(val_summary, global_step=0)
            print("Initial Loss on validation dataset: {:.6f}".format(val_loss))

        ################ TRAINING ################
        train_start = time.time()
        for step in xrange(sess.run(global_step), self.max_steps):
            start_time = time.time()
            val_str = ''
            if step % self.summaries_save_freq == 0 or step == self.max_steps - 1:
                # Train model and record summaries
                _, loss_total, summary = sess.run([train_op, loss_op, summary_op])
                summary_writer.add_summary(summary, global_step=step)
                duration = time.time() - start_time
                if self.has_val_data and step != 0:
                    # Compute validation loss
                    val_loss, val_summary = sess.run([val_loss_op, val_summary_op])
                    summary_writer.add_summary(val_summary, global_step=step)
                    val_str = ', val loss: {:.6f}'.format(val_loss)
            else: # Train only
                _, loss_total = sess.run([train_op, loss_op])
                duration = time.time() - start_time
            assert not np.isnan(loss_total), 'Model diverged with loss = NaN'

            if step % self.training_print_freq == 0 or step == self.max_steps - 1:
                examples_per_sec = self.batch_size / duration
                sec_per_batch = float(duration)
                format_str = ('{}: step {}, loss: {:.6f} ({:.1f} data/s; {:.3f} s/bch)'
                    .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total, examples_per_sec, sec_per_batch))
                print(format_str + val_str)

            if (step + 1) % self.ckpts_save_freq == 0 or step == self.max_steps - 1:
                # Save current model in a checkpoint
                self.save(sess, self.checkpoints_dir, step + 1)
        print_("Training duration: {:0.4f}s\n".format(time.time() - train_start), 'm')
        print_("--------End of training--------\n", 'm')
        # Free all resources associated with the session
        sess.close()

    def save(self, sess, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, self.ckpt_save_name), global_step=step)

    def load(self, sess, checkpoint_dir):
        ckpt_names = get_ckpt_list(checkpoint_dir)
        if not ckpt_names: # list is empty
            print_("No checkpoints found in {}\n".format(checkpoint_dir), 'm')
            return False
        else:
            print_("Found checkpoints:\n", 'm')
            for name in ckpt_names:
                print("    {}".format(name))
            # Ask user if they prefer to start training from scratch or resume training on a specific ckeckpoint 
            while True:
                mode=str(raw_input('Start training from scratch (start) or resume training from a previous checkpoint (choose one of the above): '))
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
                self.saver.restore(sess, os.path.join(checkpoint_dir, mode))
                print_("...Checkpoint {} loaded\n".format(mode), 'm')
                return True
            else:
                raise ValueError("User input is neither 'start' nor a valid checkpoint")

def parse_args():
    parser = argparse.ArgumentParser(description='Model training arguments')
    parser.add_argument('--bch', type=int, default=16, dest='batch_size', help='training batch size')
    parser.add_argument('--ep', type=int, default=10000, dest='epoch', help='training epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--seed', type=int, default=None, dest='seed', help='set random seed for deterministic training')
    parser.add_argument('--no-gpu-patch', dest='no_gpu_patch', default=False, action='store_true', help='if seed is set, add this tag for much faster but slightly less deterministic training')
    parser.add_argument('--no-resume', dest='no_resume', default=False, action='store_true',  help="start training from scratch")
    parser.add_argument('--name', type=str, default="trainingTemplateTF", dest='ckpt_save_name', help='name of saved checkpoints/model weights')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # set up model to train
    model = TrainModel(args)
    model.train()
