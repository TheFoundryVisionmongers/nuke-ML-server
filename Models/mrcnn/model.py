# The inference method is based on:
# Facebook infer_simple.py file:
# https://github.com/facebookresearch/Detectron/blob/master/tools/infer_simple.py
# Licensed under the Facebook Detectron License:
# (https://github.com/facebookresearch/Detectron/blob/master/LICENSE)
# A copy of the license can be found in the current folder "LICENSE_detectron".
#
# We are sharing this under the Apache License.
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
        self.name = 'Mask RCNN'

        # TODO: Make cfg_file and weights options
        self.cfg_file = 'models/mrcnn/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml'
        self.default_cfg = copy.deepcopy(AttrDict(cfg)) #cfg from detectron.core.config
        self.mrcnn_cfg = AttrDict()
        self.weights = 'models/mrcnn/model_final.pkl'
        self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

        # Inference options
        self.show_box = True
        self.show_class = True
        self.thresh = 0.7
        self.alpha = 0.4
        self.show_border = True
        self.border_thick = 1
        self.bbox_thick = 1
        self.font_scale = 0.35
        self.binary_masks = False

        # Define exposed options
        self.options = ('show_box', 'show_class', 'thresh', 'alpha', 'show_border',
         'border_thick', 'bbox_thick', 'font_scale', 'binary_masks')
        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def inference(self, image_list):
        """Does an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list.

        # Returns:
            The result of the inference.
        """

        image = image_list[0]
        image = self.linear_to_srgb(image)*255.
        imcpy = image.copy()

        # Switch to mrcnn worspace to keep correct blobs
        workspace.SwitchWorkspace('mrcnn_workspace', True)

        # Initialise the DLL model out of the configuration and weigts files
        if not hasattr(self, 'model'):
            # Reset to default config
            merge_cfg_from_cfg(self.default_cfg)
            # Load mask rcnn configuration file
            merge_cfg_from_file(self.cfg_file)
            assert_and_infer_cfg(cache_urls=False, make_immutable=False)
            self.model = infer_engine.initialize_model_from_cfg(self.weights)
            # Save mask rcnn full configuration file
            self.mrcnn_cfg = copy.deepcopy(AttrDict(cfg)) #cfg from detectron.core.config
        else:
            # There is a global config file for all detectron models (Densepose, Mask RCNN..)
            # Check if current global config file is correct for mask rcnn
            if not dict_equal(self.mrcnn_cfg, cfg):
                # Load mask rcnn configuration file
                merge_cfg_from_cfg(self.mrcnn_cfg)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, _ = infer_engine.im_detect_all(
                self.model, image[:, :, ::-1], None
                )

        if self.binary_masks:
            res = vis_utils.vis_one_image_binary(
                imcpy,
                cls_boxes,
                cls_segms,
                thresh=self.thresh)
        else:
            res = vis_utils.vis_one_image_opencv(
                imcpy,
                cls_boxes,
                cls_segms,
                cls_keyps,
                thresh=self.thresh,
                show_box=self.show_box,
                show_class=self.show_class,
                dataset=self.dummy_coco_dataset,
                alpha=self.alpha,
                show_border=self.show_border,
                border_thick=self.border_thick,
                bbox_thick=self.bbox_thick,
                font_scale=self.font_scale
                )

        res = self.srgb_to_linear(res.astype(np.float32) / 255.)

        return [res]