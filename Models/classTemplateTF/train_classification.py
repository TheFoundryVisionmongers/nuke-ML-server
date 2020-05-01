from __future__ import print_function

import sys
import os
import time
import random
import argparse
from datetime import datetime

import scipy.misc
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution() # For TF 2.x compatibility

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_builder import mobilenet_transfer
from common.util import im2uint8, get_filepaths_from_dir, get_saved_model_list, get_labels_from_dir, print_

class TrainModel(object):
    """Train the chosen model from the given input and groundtruth data"""

    def __init__(self, args):
        # Training hyperparameters
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_model_period = 1 # save model weights every N epochs
        # Training and validation dataset paths
        self.train_data_path = './data/train'
        self.val_data_path = './data/validation'
        # Where to save and load model weights (=checkpoints)
        self.checkpoints_dir = './checkpoints'
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.ckpt_save_name = 'classTemplate'
        # Where to save tensorboard summaries
        self.summaries_dir = './summaries'
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)

        # Get training dataset as lists of image paths
        self.train_gt_data_list = get_filepaths_from_dir(self.train_data_path)
        if len(self.train_gt_data_list) is 0:
            raise ValueError("No training data found in folder {}".format(self.train_data_path))
        elif (len(self.train_gt_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of training data = {})"
                .format(self.batch_size, len(self.train_gt_data_list)))

        # Get validation dataset if provided
        self.has_val_data = True
        self.val_gt_data_list = get_filepaths_from_dir(self.val_data_path)
        if len(self.val_gt_data_list) is 0:
            print("No validation data found in {}, 20% of training data will be used as validation data".format(self.val_data_path))
            self.has_val_data = False
            self.validation_split = 0.2
        elif (len(self.val_gt_data_list) < self.batch_size):
            raise ValueError("Batch size must be smaller than the dataset (batch size = {}, number of validation data = {})"
                .format(self.batch_size, len(self.val_gt_data_list)))
        else:
            print_("Number of validation data: {}\n".format(len(self.val_gt_data_list)), 'm')
            self.validation_split = 0.0
        
        self.train_labels = get_labels_from_dir(self.train_data_path)
        # Check class labels are the same
        if self.has_val_data:            
            self.val_labels = get_labels_from_dir(self.val_data_path)
            if self.train_labels != self.val_labels:
                if len(self.train_labels) != len(self.val_labels):
                    raise ValueError("{} and {} should have the same number of subdirectories ({}!={})"
                    .format(self.train_data_path, self.val_data_path, len(self.train_labels), len(self.val_labels)))
                raise ValueError("{} and {} should have the same subdirectory label names ({}!={})"
                    .format(self.train_data_path, self.val_data_path, self.train_labels, self.val_labels))
        
        # Compute and print training hyperparameters
        self.batch_per_epoch = int(np.ceil(len(self.train_gt_data_list) / float(self.batch_size)))
        self.max_steps = int(self.epoch * (self.batch_per_epoch))
        print_("Number of training data: {}\nNumber of batches per epoch: {} (batch size = {})\nNumber of training steps for {} epochs: {}\n"
            .format(len(self.train_gt_data_list), self.batch_per_epoch, self.batch_size, self.epoch, self.max_steps), 'm')
        print("Class labels: {}".format(self.train_labels))

    def train(self):
        # Build model
        self.model = mobilenet_transfer(len(self.train_labels))
        # Configure the model for training
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        # Print current model layers
        # self.model.summary()

        # Set preprocessing function
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # scale pixels between -1 and 1, sample-wise
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
            validation_split=self.validation_split)
        # Get classification data
        train_generator=datagen.flow_from_directory(
            self.train_data_path,
            target_size=(224,224),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training')
        if self.has_val_data:
            validation_generator=datagen.flow_from_directory(
                self.val_data_path,
                target_size=(224,224),
                color_mode='rgb',
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True)
        else: # Generate a split of the training data as validation data
            validation_generator=datagen.flow_from_directory(
            self.train_data_path, # subset from training data path
            target_size=(224,224),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='validation')
        
        # Callback for creating Tensorboard summary
        summary_name = "classif_data{}_bch{}_ep{}".format(len(self.train_gt_data_list), self.batch_size, self.epoch)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.summaries_dir, summary_name))
        # Callback for saving models periodically        
        class_labels_save = '_'.join(self.train_labels) + '.'
        # 'acc' is the training accuracy and 'val_acc' is the validation set accuracy
        self.ckpt_save_name = class_labels_save + self.ckpt_save_name + "-val_acc{val_acc:.2f}-acc{acc:.2f}-ep{epoch:04d}.h5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, self.ckpt_save_name),
            save_weights_only=False,
            period=self.save_model_period,
            save_best_only=True, monitor='val_acc', mode='max'
            )

        # Check if there are intermediate trained model to load
        # Uncomment following lines if you want to resume from a previous saved model
        # if not self.load_model():
        #     print_("Starting training from scratch\n", 'm')

        # Train the model
        fit_history = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.n // self.batch_size,
            validation_data=validation_generator,
            validation_steps= validation_generator.n // self.batch_size,
            epochs=self.epoch,            
            callbacks=[checkpoint_callback, tensorboard_callback])

        print_("--------End of training--------\n", 'm')

    def load_model(self):
        """Ask user if start training from scratch or resume from a previous checkpoint
        
        If resume, load model in self.model and return True, else return False
        """
        ckpt_names = get_saved_model_list(self.checkpoints_dir)
        if not ckpt_names: # list is empty
            print_("No checkpoints found in {}\n".format(self.checkpoint_dir), 'm')
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
                self.model = tf.keras.models.load_model(os.path.join(self.checkpoints_dir, mode))
                print_("...Checkpoint {} loaded\n".format(mode), 'm')
                return True
            else:
                raise ValueError("User input is neither 'start' nor a valid checkpoint")

def parse_args():
    parser = argparse.ArgumentParser(description='Model training arguments')
    parser.add_argument('--bch', type=int, default=16, dest='batch_size', help='training batch size')
    parser.add_argument('--ep', type=int, default=100, dest='epoch', help='training epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # set up model to train
    model = TrainModel(args)
    model.train()