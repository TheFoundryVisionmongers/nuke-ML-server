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

import tensorflow as tf

class ResNetBlock(tf.keras.layers.Layer):
    """Classic ResNet residual block"""

    def __init__(self, new_dim=32, ksize=5, name='resblock'):
        super(ResNetBlock, self).__init__(name=name)
        self.conv2D_1 = tf.keras.layers.Conv2D(
            filters=new_dim, kernel_size=ksize, padding='SAME',
            activation=tf.nn.relu, name='conv1')
        self.conv2D_2 = tf.keras.layers.Conv2D(
            filters=new_dim, kernel_size=ksize, padding='SAME',
            activation=None, name='conv2')

    def call(self, inputs):
        x = self.conv2D_1(inputs)
        x = self.conv2D_2(x)
        return x + inputs

class EncoderDecoder(tf.keras.Model):
    """Create an encoder decoder model"""

    def __init__(self, n_levels, scale, channels, name='g_net'):
        super(EncoderDecoder, self).__init__(name=name)
        self.n_levels = n_levels
        self.scale = scale

        # Encoder layers
        self.conv1_1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, padding='SAME',
            activation=tf.nn.relu, name='enc1_1')
        self.block1_2 = ResNetBlock(32, 5, name='enc1_2')
        self.block1_3 = ResNetBlock(32, 5, name='enc1_3')
        self.block1_4 = ResNetBlock(32, 5, name='enc1_4')
        self.conv2_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=5, strides=2,
            padding='SAME', activation=tf.nn.relu, name='enc2_1')
        self.block2_2 = ResNetBlock(64, 5, name='enc2_2')
        self.block2_3 = ResNetBlock(64, 5, name='enc2_3')
        self.block2_4 = ResNetBlock(64, 5, name='enc2_4')
        self.conv3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=5, strides=2,
            padding='SAME', activation=tf.nn.relu, name='enc3_1')
        self.block3_2 = ResNetBlock(128, 5, name='enc3_2')
        self.block3_3 = ResNetBlock(128, 5, name='enc3_3')
        self.block3_4 = ResNetBlock(128, 5, name='enc3_4')
        # Decoder layers
        self.deblock3_3 = ResNetBlock(128, 5, name='dec3_3')
        self.deblock3_2 = ResNetBlock(128, 5, name='dec3_2')
        self.deblock3_1 = ResNetBlock(128, 5, name='dec3_1')
        self.deconv2_4 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, strides=2,
            padding='SAME', activation=tf.nn.relu, name='dec2_4')
        self.deblock2_3 = ResNetBlock(64, 5, name='dec2_3')
        self.deblock2_2 = ResNetBlock(64, 5, name='dec2_2')
        self.deblock2_1 = ResNetBlock(64, 5, name='dec2_1')
        self.deconv1_4 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=4, strides=2,
            padding='SAME', activation=tf.nn.relu, name='dec1_4')
        self.deblock1_3 = ResNetBlock(32, 5, name='dec1_3')
        self.deblock1_2 = ResNetBlock(32, 5, name='dec1_2')
        self.deblock1_1 = ResNetBlock(32, 5, name='dec1_1')
        self.deconv0_4 = tf.keras.layers.Conv2DTranspose(
            filters=channels, kernel_size=5, padding='SAME',
            activation=None, name='dec1_0')

    def call(self, inputs, reuse=False):
        # Apply encoder decoder
        n, h, w, c = inputs.get_shape().as_list()
        n_outputs = []
        input_pred = inputs
        with tf.compat.v1.variable_scope('', reuse=reuse):
            for i in xrange(self.n_levels):
                scale = self.scale ** (self.n_levels - i - 1)
                hi = int(round(h * scale))
                wi = int(round(w * scale))
                input_init = tf.image.resize(inputs, [hi, wi], method=0)
                input_pred = tf.stop_gradient(tf.image.resize(input_pred, [hi, wi], method=0))
                input_all = tf.concat([input_init, input_pred], axis=3, name='inp')

                # Encoder
                conv1_1 = self.conv1_1(input_all)
                conv1_2 = self.block1_2(conv1_1)
                conv1_3 = self.block1_3(conv1_2)
                conv1_4 = self.block1_4(conv1_3)
                conv2_1 = self.conv2_1(conv1_4)
                conv2_2 = self.block2_2(conv2_1)
                conv2_3 = self.block2_3(conv2_2)
                conv2_4 = self.block2_4(conv2_3)
                conv3_1 = self.conv3_1(conv2_4)
                conv3_2 = self.block3_2(conv3_1)
                conv3_3 = self.block3_3(conv3_2)
                encoded = self.block3_4(conv3_3)

                # Decoder
                deconv3_3 = self.deblock3_3(encoded)
                deconv3_2 = self.deblock3_2(deconv3_3)
                deconv3_1 = self.deblock3_1(deconv3_2)
                deconv2_4 = self.deconv2_4(deconv3_1)
                cat2 = deconv2_4 + conv2_4 # Skip connection
                deconv2_3 = self.deblock2_3(cat2)
                deconv2_2 = self.deblock2_2(deconv2_3)
                deconv2_1 = self.deblock2_1(deconv2_2)
                deconv1_4 = self.deconv1_4(deconv2_1)
                cat1 = deconv1_4 + conv1_4 # Skip connection
                deconv1_3 = self.deblock1_3(cat1)
                deconv1_2 = self.deblock1_2(deconv1_3)
                deconv1_1 = self.deblock1_1(deconv1_2)
                input_pred = self.deconv0_4(deconv1_1)

                if i >= 0:
                    n_outputs.append(input_pred)
                if i == 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
        return n_outputs

def mobilenet_transfer(class_number):
    """Return a classification model with a mobilenet backbone pretrained on ImageNet
    
    # Arguments:
        class_number: Number of classes / labels to detect
    """
    # Import the mobilenet model and discards the last 1000 neuron layer.
    base_model = tf.keras.applications.MobileNet(input_shape=(224,224,3), weights='imagenet',include_top=False, pooling='avg')

    x = base_model.output
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dense(512,activation='relu')(x)
    # Final layer with softmax activation
    preds = tf.keras.layers.Dense(class_number,activation='softmax')(x)
    # Build the model
    model = tf.keras.models.Model(inputs=base_model.input,outputs=preds)

    # Freeze base_model
    # for layer in base_model.layers: # <=> to [:86]
    #     layer.trainable = False
    # Freeze the first 60 layers and fine-tune the rest
    for layer in model.layers[:60]:
        layer.trainable=False
    for layer in model.layers[60:]:
        layer.trainable=True

    return model

def baseline_model(input_shape, output_param_number=1, hidden_layer_size=16):
    """Return a fully connected model with 1 hidden layer"""
    if hidden_layer_size < output_param_number:
        raise ValueError("Neurons in the hidden layer (={}) \
            should be > output param number (={})".format(
                hidden_layer_size, output_param_number))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    # Regular densely connected NN layer
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(output_param_number, activation=None)) # linear activation
    return model