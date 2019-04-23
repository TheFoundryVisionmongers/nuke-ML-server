# Copyright (c) 2019 The Foundry Visionmongers Ltd. All Rights Reserved.
# This is strictly non-commercial.

from ..baseModel import BaseModel

import cv2
import numpy as np

import message_pb2

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Gaussian Blur'

        self.kernel_size = 5
        self.make_blur = False

        # Define options
        self.options = ('kernel_size',)
        self.buttons = ('make_blur',)

        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def inference(self, image_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list

        Return the result of the inference.
        """
        image = image_list[0]
        image = self.linear_to_srgb(image)
        image = (image * 255).astype(np.uint8)
        kernel = self.kernel_size * 2 + 1
        blur = cv2.GaussianBlur(image, (kernel, kernel), 0)
        blur = blur.astype(np.float32) / 255.
        blur = self.srgb_to_linear(blur)
        
        # If make_blur button is pressed in Nuke
        if self.make_blur:
            script_msg = message_pb2.FieldValuePairAttrib()
            script_msg.name = "PythonScript"
            # Create a Python script message to run in Nuke
            python_script = self.blur_script(blur)
            script_msg_val = script_msg.values.add()
            script_msg_str = script_msg_val.string_attributes.add()
            script_msg_str.values.extend([python_script])
            return [blur, script_msg]

        return [blur]

    def blur_script(self, image):
        """Return the Python script function to create a pop up window in Nuke.

        The pop up window displays the brightest pixel position of the given image.
        """
        # Compute brightest pixel of the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        [min_val, max_val, min_loc, max_loc] = cv2.minMaxLoc(gray)
        # Y axis are inversed in Nuke
        max_loc = (max_loc[0], image.shape[0] - max_loc[1])
        popup_msg = (
            "Brightest pixel of the blurred image\\n"
            "Location: {}, Value: {:.3f}."
            ).format(max_loc, max_val)
        script = "nuke.message('{}')\n".format(popup_msg)
        return script