import sys
import os
import re

import numpy as np
import OpenEXR, Imath
import PIL.Image as Image # PIL (Python Imaging Library)

import tensorflow as tf

def print_(str, colour='', bold=False):
    if colour == 'w': # yellow warning
        sys.stdout.write('\033[93m')
    elif colour == "e": # red error
        sys.stdout.write('\033[91m')
    elif colour == "m": # magenta info
        sys.stdout.write('\033[95m')
    if bold:
        sys.stdout.write('\033[1m')
    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

def get_filepaths_from_dir(dir_path):
    """Recursively walk through the given directory and return a list of file paths
    """
    data_list = []
    for (root, directories, filenames) in os.walk(dir_path):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            data_list += [os.path.join(root,filename)]
    return data_list

def get_labels_from_dir(dir_path):
    """Return classification class labels (= first subdirectories names)
    """
    labels_list = []
    for (root, directories, filenames) in os.walk(dir_path):
        directories.sort()
        labels_list += directories
        # Break to only keep the top directory
        break
    # Remove '.' in folder names for label retrieval in model.py
    labels_list = [''.join(label.split('.')) for label in labels_list]
    return labels_list

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """Use mylist.sort(key=natural_keys) to sort mylist in human order
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_saved_model_list(ckpt_dir):
    """Return a list of HDF5 models found in ckpt_dir
    """
    filenames_list = []
    for (root, directories, filenames) in os.walk(ckpt_dir):
        filenames_list += filenames
        # Break to only keep the top directory
        break
    ckpt_list = []
    for filename in filenames_list:
        if filename.endswith(('.h5', '.hdf5')):
            ckpt_list += [filename]
    ckpt_list.sort(key=natural_keys)
    return ckpt_list

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)