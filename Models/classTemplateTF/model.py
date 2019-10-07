from __future__ import print_function

import sys
import os
import time

import scipy.misc
import numpy as np
import cv2

import tensorflow as tf

from ..baseModel import BaseModel
from ..common.model_builder import mobilenet_transfer
from ..common.util import print_, get_saved_model_list

import message_pb2

class Model(BaseModel):
    """Load your trained model and do inference in Nuke"""

    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Classification Template'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.checkpoints_dir = os.path.join(dir_path, 'checkpoints')
        self.batch_size = 1

        # Initialise checkpoint name to the most recent trained model
        ckpt_names = get_saved_model_list(self.checkpoints_dir)
        if not ckpt_names: # empty list
            self.checkpoint_name = ''
        else:
            self.checkpoint_name = ckpt_names[-1]
        self.prev_ckpt_name = self.checkpoint_name

        # Button to get classification label
        self.get_label = False

        # Define options
        self.options = ('checkpoint_name',)
        self.buttons = ('get_label',)
        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def load_model(self):
        # Check if empty or invalid checkpoint name
        if self.checkpoint_name=='':
            ckpt_names = get_saved_model_list(self.checkpoints_dir)
            if not ckpt_names:
                raise ValueError("No checkpoints found in {}".format(self.checkpoints_dir))
            else:
                raise ValueError("Empty checkpoint name, try an available checkpoint in {} (ex: {})"
                    .format(self.checkpoints_dir, ckpt_names[-1]))
        print_("Loading trained model checkpoint...\n", 'm')
        # Load from given checkpoint file name
        model = tf.keras.models.load_model(os.path.join(self.checkpoints_dir, self.checkpoint_name))
        print_("...Checkpoint {} loaded\n".format(self.checkpoint_name), 'm')
        return model

    def inference(self, image_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list

        Return the result of the inference.
        """
        image = image_list[0]
        image = self.linear_to_srgb(image).copy()
        image = (image * 255).astype(np.uint8)

        if not hasattr(self, 'model'):
            # Initialise tensorflow graph
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess=tf.Session(config=config)
            # Load most recent trained model
            self.model = self.load_model()
            self.class_labels = (self.checkpoint_name.split('.')[0]).split('_')

        # If checkpoint name has changed, load new checkpoint
        if self.prev_ckpt_name != self.checkpoint_name or self.checkpoint_name == '':
            self.model = self.load_model()
            self.class_labels = (self.checkpoint_name.split('.')[0]).split('_')
            # If checkpoint correctly loaded, update previous checkpoint name
            self.prev_ckpt_name = self.checkpoint_name

        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
        # Predict on new data
        image_batch = np.expand_dims(image, 0)
        # Preprocess a numpy array encoding a batch of images (RGB values within [0, 255])
        image_batch = tf.keras.applications.mobilenet.preprocess_input(image_batch)
        start = time.time()
        y_prob = self.model.predict(image_batch)
        y_class = y_prob.argmax(axis=-1)[0]
        duration = time.time() - start
        # Print results on server side
        print('Inference duration: {:4.3f}s'.format(duration))
        class_scores = str(["{0:0.4f}".format(i) for i in y_prob[0]]).replace("'", "")
        print("Class scores: {} --> Label: {}".format(class_scores, self.class_labels[y_class]))

        # If get_label button is pressed in Nuke
        if self.get_label:
            # Send back which class was detected
            script_msg = message_pb2.FieldValuePairAttrib()
            script_msg.name = "PythonScript"
            # Create a Python script message to run in Nuke
            nuke_msg = "Class scores: {}\\nLabel: {}".format(class_scores, self.class_labels[y_class])
            python_script = "nuke.message('{}')\n".format(nuke_msg)
            script_msg_val = script_msg.values.add()
            script_msg_str = script_msg_val.string_attributes.add()
            script_msg_str.values.extend([python_script])
            return [image_list[0], script_msg]
        return [image_list[0]]