# Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.
#
# Based on:
# Facebook vis.py file:
# https://github.com/facebookresearch/DensePose/blob/master/detectron/utils/vis.py
# Licensed under the Facebook Densepose License:
# (https://github.com/facebookresearch/DensePose/blob/master/LICENSE)
# A copy of the license can be found in the current folder "LICENSE_densepose",
#
# We are sharing this under the "Attribution-NonCommercial 4.0 International"
########################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

import detectron.utils.vis as vis_utils

_WHITE = (255, 255, 255)

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

def vis_isocontour(img, img_IUV):
    """Visualizes the isocontour of DensePose IUV fields results using matplotlib

    # Arguments:
        img: initial image
        img_IUV: densepose UV fields inference of img
        
    # Returns:
        A numpy array with isocontour following the given IUV fields
    """

    # Matplotlib visualisation of isocontour
    fig = plt.figure(figsize=[12,12])
    plt.contour( img_IUV[:,:,1]/255.,10, linewidths = 1 ) # isocontour of U
    plt.contour( img_IUV[:,:,2]/255.,10, linewidths = 1 ) # isocontour of V

    # Get the image content only
    fig.subplots_adjust(bottom = 0, top = 1, right = 1, left = 0)
    plt.axis('off')
    plt.imshow(img /255.) #matplotlib wants floats between 0 and 1 or integer between 0 and 255

    # Save matplotlib figure and load it as a cv array
    fig_name = '/tmp/isocontour.jpg'
    plt.savefig(fig_name)
    plt.close()

    img_plt = cv2.imread(fig_name)
    # Resize and crop image to fit initial image size
    size_max = max(img.shape[1], img.shape[0])
    img_plt_resized= cv2.resize(img_plt[:,:,::-1], (size_max, size_max))
    diff_heigt = abs(img.shape[0] - img_plt_resized.shape[0])/2
    diff_width = abs(img.shape[1] - img_plt_resized.shape[1])/2
    img_plt_cropped = img_plt_resized[diff_heigt:img_plt_resized.shape[0]-diff_heigt,
        diff_width:img_plt_resized.shape[1]-diff_width].copy()

    return img_plt_cropped


def vis_densepose(
    img, cls_boxes, cls_bodys, show_human_index=False, show_uv=True, show_grid=False,
    show_border=False, border_thick=1, alpha=0.4):
    """Constructs a numpy array showing the densepose detection

    # Arguments:
        img: image used for densepose inference
        cls_boxes: bounding boxes found during inference of image img
        cls_bodys: UV bodys found during inference of image img
        show_uv: show the UV fields
        show_grid: show the isocontours of the UV fields
        alpha: how much blended the densepose visualisation is
            with the original image img
        
    # Returns:
        A numpy image array showing the densepose detection
    """

    if isinstance(cls_boxes, list):
        boxes, _, _, _ = vis_utils.convert_from_cls_format(
            cls_boxes, None, None)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < 0.9:
        return img

    ##  Get full IUV image out 
    IUV_fields = cls_bodys[1]
    #
    all_coords = np.zeros(img.shape) # I, U and V channels
    all_inds = np.zeros([img.shape[0],img.shape[1]]) # all_inds stores index of corresponding human (background=0)
    ##
    inds = np.argsort(boxes[:,4])
    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind,:]
        if entry[4] > 0.65: #second threshold on human proba to be in box
            entry=entry[0:4].astype(int)
            ####
            output = IUV_fields[ind]
            ####
            all_coords_box = all_coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]
            all_coords_box[all_coords_box==0]=output.transpose([1,2,0])[all_coords_box==0]
            all_coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]= all_coords_box
            ###
            CurrentMask = (output[0,:,:]>0).astype(np.float32)
            all_inds_box = all_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]]
            all_inds_box[all_inds_box==0] = CurrentMask[all_inds_box==0]*(i+1)
            all_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]] = all_inds_box
    #
    all_coords[:,:,1:3] = 255. * all_coords[:,:,1:3]
    all_coords[all_coords>255] = 255.
    all_coords = all_coords.astype(np.uint8)
    all_inds = all_inds.astype(np.uint8)

    res = img #initialise image result to input image

    if show_human_index:
        all_inds_vis = all_inds * (210.0/all_inds.max()) # normalise all_inds values between 0. and 210.
        all_inds_stacked = np.stack((all_inds_vis,)*3, axis = -1)
        res = all_inds_stacked
    
    elif show_grid:
        res = vis_isocontour(img, all_coords)
        alpha = 0.
        
    elif show_uv:
        res = all_coords # I, U and V channels
        alpha = 0.

    if show_border:
        res = vis_mask(res, all_inds, np.array([150., 20., 200.]),
            alpha=alpha, show_border=show_border, border_thick=border_thick)

    return res
