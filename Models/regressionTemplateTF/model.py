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

from __future__ import print_function

import sys
import os
import time

import scipy.misc
import numpy as np
import cv2

import tensorflow as tf
tf.compat.v1.disable_eager_execution() # For TF 2.x compatibility

from models.baseModel import BaseModel
from models.common.model_builder import baseline_model
from models.common.util import print_, get_ckpt_list, linear_to_srgb, srgb_to_linear

import message_pb2

class Model(BaseModel):
    """Load your trained model and do inference in Nuke"""

    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Regression Template TF'
        self.n_levels = 3
        self.scale = 0.5
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.checkpoints_dir = os.path.join(dir_path, 'checkpoints')
        self.patch_size = 50
        self.output_param_number = 1

        # Initialise checkpoint name to the latest checkpoint
        ckpt_names = get_ckpt_list(self.checkpoints_dir)
        if not ckpt_names: # empty list
            self.checkpoint_name = ''
        else:
            latest_ckpt = tf.compat.v1.train.latest_checkpoint(self.checkpoints_dir)
            if latest_ckpt is not None:
                self.checkpoint_name = latest_ckpt.split('/')[-1]
            else:
                self.checkpoint_name = ckpt_names[-1]
        self.prev_ckpt_name = self.checkpoint_name
        
        # Silence TF log when creating tf.Session()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Define options
        self.gamma_to_predict = 1.0
        self.predict = False
        self.options = ('checkpoint_name', 'gamma_to_predict',)
        self.buttons = ('predict',)
        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def load(self, model):
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
        self.saver.restore(self.sess, os.path.join(self.checkpoints_dir, self.checkpoint_name))
        print_("...Checkpoint {} loaded\n".format(self.checkpoint_name), 'm')

    def inference(self, image_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list

        Return the result of the inference.
        """
        image = image_list[0]
        image = linear_to_srgb(image).copy()

        if not hasattr(self, 'sess'):
            # Initialise tensorflow graph
            tf.compat.v1.reset_default_graph()
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess=tf.compat.v1.Session(config=config)
            # Input is stacked histograms of original and gamma-graded images.
            input_shape = [1, 2, 100]
            # Initialise input placeholder size
            self.input = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
            self.model = baseline_model(
                input_shape=input_shape[1:],
                output_param_number=self.output_param_number)
            self.infer_op = self.model(self.input)
            # Load latest model checkpoint
            self.saver = tf.compat.v1.train.Saver()
            self.load(self.model)
            self.prev_ckpt_name = self.checkpoint_name

        # If checkpoint name has changed, load new checkpoint
        if self.prev_ckpt_name != self.checkpoint_name or self.checkpoint_name == '':
            self.load(self.model)
            # If checkpoint correctly loaded, update previous checkpoint name
            self.prev_ckpt_name = self.checkpoint_name

        # Preprocess image same way we preprocessed it for training
        # Here for gamma correction compute histograms
        def histogram(x, value_range=[0.0, 1.0], nbins=100):
            """Return histogram of tensor x"""
            h, w, c = x.shape
            hist = tf.histogram_fixed_width(x, value_range, nbins=nbins)
            hist = tf.divide(hist, h * w * c)
            return hist
        with tf.compat.v1.Session() as sess:
            # Convert to grayscale
            img_gray = tf.image.rgb_to_grayscale(image)
            img_gray = tf.image.resize(img_gray, [self.patch_size, self.patch_size])
            # Apply gamma correction
            img_gray_grade = tf.math.pow(img_gray, self.gamma_to_predict)
            img_grade = tf.math.pow(image, self.gamma_to_predict)
            # Compute histograms
            img_hist = histogram(img_gray)
            img_grade_hist = histogram(img_gray_grade)
            hists_op = tf.stack([img_hist, img_grade_hist], axis=0)
            hists, img_grade = sess.run([hists_op, img_grade])
            res_img = srgb_to_linear(img_grade)

        hists_batch = np.expand_dims(hists, 0)
        start = time.time()
        # Run model inference
        inference = self.sess.run(self.infer_op, feed_dict={self.input: hists_batch})
        duration = time.time() - start
        print('Inference duration: {:4.3f}s'.format(duration))
        res = inference[-1]
        print("Predicted gamma: {}".format(res))

        # If predict button is pressed in Nuke
        if self.predict:
            script_msg = message_pb2.FieldValuePairAttrib()
            script_msg.name = "PythonScript"
            # Create a Python script message to run in Nuke
            python_script = self.nuke_script(res)
            script_msg_val = script_msg.values.add()
            script_msg_str = script_msg_val.string_attributes.add()
            script_msg_str.values.extend([python_script])
            return [res_img, script_msg]

        return [res_img]

    def nuke_script(self, res):
        """Return the Python script function to create a pop up window in Nuke."""
        popup_msg = "Predicted gamma: {}".format(res)
        script = "nuke.message('{}')\n".format(popup_msg)
        return script