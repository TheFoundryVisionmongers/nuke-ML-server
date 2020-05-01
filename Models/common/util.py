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

import sys
import os
import re

import numpy as np
import OpenEXR, Imath
import cv2

import tensorflow as tf
tf.compat.v1.disable_eager_execution() # For TF 2.x compatibility

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

## GET DATA ##

def get_filepaths_from_dir(dir_path):
    """Recursively walk through the given directory and return a list of file paths"""
    data_list = []
    for (root, directories, filenames) in os.walk(dir_path):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            data_list += [os.path.join(root,filename)]
    return data_list

def get_labels_from_dir(dir_path):
    """Return classification class labels (= first subdirectories names)"""
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
    """Use mylist.sort(key=natural_keys) to sort mylist in human order"""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_ckpt_list(ckpt_dir):
    filenames_list = []
    for (root, directories, filenames) in os.walk(ckpt_dir):
        filenames_list += filenames
        # Break to only keep the top directory
        break
    ckpt_list = []
    for filename in filenames_list:
        split = filename.split('.')
        if len(split) > 1 and split[-1] == 'index':
            # remove .index to get the ckeckpoint name
            ckpt_list += [filename[:-6]]
    ckpt_list.sort(key=natural_keys)
    return ckpt_list

def get_saved_model_list(ckpt_dir):
    """Return a list of HDF5 models found in ckpt_dir"""
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

## PROCESS DATA ##

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

def srgb_to_linear(x):
    """Transform the image from sRGB to linear"""
    a = 0.055
    x = np.clip(x, 0, 1)
    mask = x < 0.04045
    x[mask] /= 12.92
    x[mask!=True] = np.exp(2.4 * (np.log(x[mask!=True] + a) - np.log(1 + a)))
    return x

def linear_to_srgb(x):
    """Transform the image from linear to sRGB"""
    a = 0.055
    x = np.clip(x, 0, 1)
    mask = x <= 0.0031308
    x[mask] *= 12.92
    x[mask!=True] = np.exp(np.log(1 + a) + (1/2.4) * np.log(x[mask!=True])) - a
    return x

## EXR DATA UTILS ##

"""
EXR utility functions have to be wrapped in a TensorFlow graph by using
tf.numpy_function(). This function requires a specific fixed return type,
which is why all EXR reading functions are of return type float32.
"""
# Imath.PixelType can have UINT unint32, HALF float16, FLOAT float32
EXR_PIX_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)
EXR_NP_TYPE = np.float32

def is_exr(filename):
    file_extension = os.path.splitext(filename)[1][1:]
    if file_extension in ['exr', 'EXR']:
        return True
    elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']:
        return False
    else:
        raise TypeError("{} unhandled type extensions. Should be one of "
            "['jpg', 'jpeg', 'png', 'bmp', 'exr']". format(file_extension))

def check_exr(exr_files, channel_names=['R', 'G', 'B']):
    """Check that exr_files (a list of EXR file(s)) have the requested channels
    and have the same data window size. Return image width and height.
    """
    if not list(channel_names):
        raise ValueError("channel_names is empty")
    if isinstance(exr_files, OpenEXR.InputFile):    # single exr file
        exr_files = [exr_files]
    elif not isinstance(exr_files, list):
        raise TypeError("type(exr_files): {}, should be str or list".format(type(exr_files)))
    # Check data window size
    data_windows = [str(exr.header()['dataWindow']) for exr in exr_files]
    if any(dw != data_windows[0] for dw in data_windows):
        raise ValueError("input and groundtruth .exr images have different size")
    # Check channel to read are present in given exr file(s)
    channels_headers = [exr.header()['channels'] for exr in exr_files]
    for channels in channels_headers:
        if any(c not in list(channels.keys()) for c in channel_names):
            raise ValueError("Try to read channels {} of an exr image with channels {}"
                .format(channel_names, list(channels.keys())))
    # Compute the size
    dw = exr_files[0].header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    return width, height

def read_exr(exr_path, channel_names=['R', 'G', 'B']):
    """Read requested channels of an exr and return them in a numpy array
    """
    # Open and check the input file
    exr_file = OpenEXR.InputFile(exr_path)
    width, height = check_exr(exr_file, channel_names)
    # Copy channels from an exr file into a numpy array
    exr_numpy = [np.frombuffer(exr_file.channel(c, EXR_PIX_TYPE), dtype=EXR_NP_TYPE)
        .reshape(height, width) for c in channel_names]
    exr_numpy = np.stack(exr_numpy, axis=-1)
    return exr_numpy

def read_resize_exr(exr_path, patch_size, channel_names=['R', 'G', 'B']):
    """Read requested channels of an exr as numpy array
    and return them resized to (patch_size, patch_size)
    """
    exr = read_exr(exr_path, channel_names)
    exr_resize = cv2.resize(exr, dsize=(patch_size, patch_size))
    return exr_resize

def read_crop_exr(exr_file, size, crop_w, crop_h, crop_size=256, channel_names=['R', 'G', 'B']):
    """Read requested channels of an exr file, crop it and return it as numpy array

    The cropping box has a size of crop_size and its bottom left point is (crop_h, crop_w)
    """
    # Read only the crop scanlines, not the full EXR image
    cnames = ''.join(channel_names)
    channels = exr_file.channels(cnames=cnames, pixel_type=EXR_PIX_TYPE,
        scanLine1=crop_h, scanLine2=crop_h + crop_size - 1)
    exr_crop = np.zeros([crop_size, crop_size, len(channel_names)], dtype=EXR_NP_TYPE)
    for idx, c in enumerate(channel_names):
        exr_crop[:,:,idx] = (np.frombuffer(channels[idx], dtype=EXR_NP_TYPE)
            .reshape(crop_size, size[0])[:, crop_w:crop_w+crop_size])
    return exr_crop

def read_crop_exr_pair(exr_path_in, exr_path_gt, crop_size=256, channel_names=['R', 'G', 'B']):
    """Read requested channels of input and groundtruth .exr image paths
    and return the same random crop of both
    """
    # Open the input file
    exr_file_in = OpenEXR.InputFile(exr_path_in)
    exr_file_gt = OpenEXR.InputFile(exr_path_gt)
    width, height = check_exr([exr_file_in, exr_file_gt], channel_names)
    # Check exr image width and height >= crop_size
    if height < crop_size or width < crop_size:
        raise ValueError("Input images size should be superior or equal to crop_size: {} < ({},{})"
            .format((width, height), crop_size, crop_size))
    # Get random crop value
    randw = np.random.randint(0, width-crop_size) if width-crop_size > 0 else 0
    randh = np.random.randint(0, height-crop_size) if height-crop_size > 0 else 0 
    # Get the crop of input and groundtruth .exr images
    exr_crop_in = read_crop_exr(exr_file_in, (width, height), randw, randh, crop_size, channel_names)
    exr_crop_gt = read_crop_exr(exr_file_gt, (width, height), randw, randh, crop_size, channel_names)
    return [exr_crop_in, exr_crop_gt]