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

import scipy.misc
import numpy as np
import cv2

import tensorflow as tf

from ..baseModel import BaseModel
from ..common.model_builder import EncoderDecoder
from ..common.util import print_, get_ckpt_list

class Model(BaseModel):
    """Load your trained model and do inference in Nuke"""

    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Training Template TF'
        self.n_levels = 3
        self.scale = 0.5
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.checkpoints_dir = os.path.join(dir_path, 'checkpoints')
        self.batch_size = 1

        # Initialise checkpoint name to the most advanced checkpoint (highest step)
        ckpt_names = get_ckpt_list(self.checkpoints_dir)
        if not ckpt_names: # empty list
            self.checkpoint_name = ''
        else:
            ckpt_steps = [int(name.split('-')[-1]) for name in ckpt_names]
            self.checkpoint_name = ckpt_names[ckpt_steps.index(max(ckpt_steps))]
        self.prev_ckpt_name = self.checkpoint_name

        self.options = ('checkpoint_name',)
        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def load(self, sess, checkpoint_dir):
        # Check if empty or invalid checkpoint name
        if self.checkpoint_name=='':
            ckpt_names = get_ckpt_list(self.checkpoints_dir)
            if not ckpt_names:
                raise ValueError("No checkpoints found in {}".format(self.checkpoints_dir))
            else:
                raise ValueError("Empty checkpoint name, try an available checkpoint in {} (ex: {})"
                    .format(self.checkpoints_dir, ckpt_names[-1]))
        print_("Loading trained model checkpoint...\n", 'm')
        # Load from given checkpoint file name
        self.saver.restore(sess, os.path.join(checkpoint_dir, self.checkpoint_name))
        print_("...Checkpoint {} loaded\n".format(self.checkpoint_name), 'm')

    def inference(self, image_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list

        Return the result of the inference.
        """
        image = image_list[0]
        image = self.linear_to_srgb(image).copy()
        H, W, channels = image.shape

        # Add padding so that width and height of image are a multiple of 16
        new_H = int(H + 16 - H%16) if H%16!=0 else H
        new_W = int(W + 16 - W%16) if W%16!=0 else W
        img_pad = np.pad(image, ((0, new_H - H), (0, new_W - W), (0, 0)), 'reflect')

        if not hasattr(self, 'sess'):
            # Initialise input placeholder size
            self.curr_height = new_H; self.curr_width = new_W
            # Initialise tensorflow graph
            tf.compat.v1.reset_default_graph()
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess=tf.compat.v1.Session(config=config)
            self.input = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, new_H, new_W, channels])
            self.model = EncoderDecoder(self.n_levels, self.scale, channels)
            self.infer_op = self.model(self.input, reuse=False)
            # Load model checkpoint having the longest training (highest step)
            self.saver = tf.compat.v1.train.Saver()
            self.load(self.sess, self.checkpoints_dir)
            self.prev_ckpt_name = self.checkpoint_name

        elif self.curr_height != new_H or self.curr_width != new_W:
            # Modify input placeholder size
            self.input = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, new_H, new_W, channels])
            self.infer_op = self.model(self.input, reuse=False)
            # Update image height and width
            self.curr_height = new_H; self.curr_width = new_W

        # If checkpoint name has changed, load new checkpoint
        if self.prev_ckpt_name != self.checkpoint_name or self.checkpoint_name == '':
            self.load(self.sess, self.checkpoints_dir)
            # If checkpoint correctly loaded, update previous checkpoint name
            self.prev_ckpt_name = self.checkpoint_name

        # Apply current model to the padded input image
        image_batch = np.expand_dims(img_pad, 0)
        start = time.time()
        # The network is expecting image_batch to be of type tf.float32
        inference = self.sess.run(self.infer_op, feed_dict={self.input: image_batch})
        duration = time.time() - start
        print('Inference duration: {:4.3f}s'.format(duration))
        res = inference[-1]
        # Remove first dimension and padding
        res = res[0, :H, :W, :]

        output_image = self.srgb_to_linear(res)
        return [output_image]