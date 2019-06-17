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
    """Recursively walk through the given directory and return a list of file paths"""
    data_list = []
    for (root, directories, filenames) in os.walk(dir_path):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            data_list += [os.path.join(root,filename)]
    return data_list

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''Use mylist.sort(key=natural_keys) to sort mylist in human order'''
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

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

def is_exr(filename):
    file_extension = os.path.splitext(filename)[1][1:]
    if file_extension in ['exr', 'EXR']:
        return True
    elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']:
        return False
    else:
        raise TypeError("{} unhandled type extensions. Should be one of "
            "['jpg', 'jpeg', 'png', 'bmp', 'exr']". format(file_extension))

def read_exr(exr_filename):
    # Open the input file
    exr_file = OpenEXR.InputFile(exr_filename)
    # Compute the size
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Read the three colour channels as 32-bit floats
    # Imath.PixelType can have UINT unint32, HALF float16, FLOAT float32
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = exr_file.channels("RGB", FLOAT)
    I = np.zeros([size[1], size[0], 3], dtype=np.float32)
    I[:,:,0] = np.frombuffer(R, dtype=np.float32).reshape(size[1], size[0])
    I[:,:,1] = np.frombuffer(G, dtype=np.float32).reshape(size[1], size[0])
    I[:,:,2] = np.frombuffer(B, dtype=np.float32).reshape(size[1], size[0])
    return I

def read_crop_exr(exr_img, size, crop_size, crop_h, crop_w):
    """Read RGB channels from an exr_img, crop it and return it as numpy array
    
    The cropping box has a size of crop_size and its bottom left point is (crop_h, crop_w)
    """
    # Imath.PixelType can have UINT unint32, HALF float16, FLOAT float32
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    # Read the three colour channels as type pt
    (R,G,B) = exr_img.channels(cnames="RGB", pixel_type=pt, scanLine1=crop_h, scanLine2=crop_h + crop_size - 1)
    I = np.zeros([crop_size, crop_size, 3], dtype=np.float32)
    I[:,:,0] = np.frombuffer(R, dtype=np.float32).reshape(crop_size, size[0])[:, crop_w:crop_w+crop_size]
    I[:,:,1] = np.frombuffer(G, dtype=np.float32).reshape(crop_size, size[0])[:, crop_w:crop_w+crop_size]
    I[:,:,2] = np.frombuffer(B, dtype=np.float32).reshape(crop_size, size[0])[:, crop_w:crop_w+crop_size]
    return I

def read_crop_exr_pair(exr_path_in, exr_path_gt, crop_size=256):
    """Read RGB channels of input and groundtruth .exr image paths, return a random crop of both"""
    # Open the input file
    exr_img_in = OpenEXR.InputFile(exr_path_in)
    exr_img_gt = OpenEXR.InputFile(exr_path_gt)
    # Check data window size
    if str(exr_img_in.header()['dataWindow']) != str(exr_img_gt.header()['dataWindow']):
        raise ValueError("input and groundtruth .exr images have different size")
    dw = exr_img_in.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Get random crop value
    randh, randw = (np.random.randint(0, size[1]-crop_size), np.random.randint(0, size[0]-crop_size))
    # Get the crop of input and groundtruth .exr images
    exr_crop_in = read_crop_exr(exr_img_in, size, crop_size, randh, randw)
    exr_crop_gt = read_crop_exr(exr_img_gt, size, crop_size, randh, randw)    
    return [exr_crop_in, exr_crop_gt]