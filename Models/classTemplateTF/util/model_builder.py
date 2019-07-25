import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications import MobileNet

def mobilenet_transfer(class_number):
    """Return a classification model with a mobilenet backbone pretrained on ImageNet
    
    # Arguments:
        class_number: Number of classes / labels to detect
    """
    # Import the mobilenet model and discards the last 1000 neuron layer.
    base_model = MobileNet(input_shape=(224,224,3), weights='imagenet',include_top=False, pooling='avg')

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
    # Freeze the first 20 layers and fine-tune the rest
    for layer in model.layers[:60]:
        layer.trainable=False
    for layer in model.layers[60:]:
        layer.trainable=True

    return model