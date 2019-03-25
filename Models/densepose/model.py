# Copyright (c) 2019 The Foundry Visionmongers Ltd.  All Rights Reserved.
#
# The inference method is based on:
# Facebook infer_simple.py file:
# https://github.com/facebookresearch/DensePose/blob/master/tools/infer_simple.py
# Licensed under the Facebook Densepose License:
# (https://github.com/facebookresearch/DensePose/blob/master/LICENSE)
# A copy of the license can be found in the current folder "LICENSE_densepose",
#
# We are sharing this under the "Attribution-NonCommercial 4.0 International"
########################################################################

from ..baseModel import BaseModel

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file, merge_cfg_from_cfg
from detectron.utils.collections import AttrDict
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils

import vis as vis_utils
from utils import dict_equal

import numpy as np
import copy

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'DensePose'

        # TODO: Make cfg_file and weights options
        self.cfg_file= 'models/densepose/DensePose_ResNet101_FPN_s1x-e2e.yaml'
        self.default_cfg = copy.deepcopy(AttrDict(cfg)) #cfg from detectron.core.config
        self.densepose_cfg = AttrDict()
        self.weights = 'models/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl'
        self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

        # Inference options
        self.show_human_index = False
        self.show_uv = False
        self.show_grid = True
        self.show_border = False
        self.border_thick = 1
        self.alpha = 0.4

        # Define exposed options
        self.options= ('show_human_index', 'show_uv', 'show_grid', 'show_border',
            'border_thick', 'alpha')
		# Define inputs/outputs
        self.inputs= {'input': 3} #3-channel input
        self.outputs= {'output': 3} #3-channel output


    def inference(self, image_list):
        """Does an inference of the DensePose model with a set of image inputs

        # Arguments:
            image_list: The inpput image list
        
        # Returns:
            The result of the inference
        """

        # Directly returns image when no inference options
        if not (self.show_human_index or self.show_uv or self.show_border or self.show_grid):
            return [image_list[0]]

        image = image_list[0]
        image = self.linear_to_srgb(image)*255.
        imcpy = image.copy()

        # Switch to densepose workspace to keep correct blobs
        workspace.SwitchWorkspace('densepose_workspace', True)

        # Initialise the DLL model out of the configuration and weigts files
        if not hasattr(self, 'model'):
            # Reset to default config
            merge_cfg_from_cfg(self.default_cfg)
            # Load densepose configuration file
            merge_cfg_from_file(self.cfg_file)
            assert_and_infer_cfg(cache_urls=False, make_immutable=False)
            self.model = infer_engine.initialize_model_from_cfg(self.weights)
            # Save densepose full configuration file
            self.densepose_cfg = copy.deepcopy(AttrDict(cfg)) #cfg from detectron.core.config
        else:
            # There is a global config file for all detectron models (Densepose, Mask RCNN..)
            # Check if current global config file is correct for densepose
            if not dict_equal(self.densepose_cfg, cfg):
                # Load densepose configuration file
                merge_cfg_from_cfg(self.densepose_cfg)

        # Compute the image inference
        with c2_utils.NamedCudaScope(0):
            # image in BGR format for inference
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                self.model, image[:, :, ::-1], None
                )

        res = vis_utils.vis_densepose(
            imcpy, # image in RGB format for visualization
            cls_boxes,
            cls_bodys,
            show_human_index=self.show_human_index,
            show_uv=self.show_uv,
            show_grid=self.show_grid,
            show_border=self.show_border,
            border_thick=self.border_thick,
            alpha=self.alpha)

        res = self.srgb_to_linear(res.astype(np.float32) / 255.)

        return [res]