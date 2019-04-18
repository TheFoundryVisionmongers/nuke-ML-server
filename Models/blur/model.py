# Copyright (c) 2019 The Foundry Visionmongers Ltd. All Rights Reserved.
# This is strictly non-commercial.

from ..baseModel import BaseModel

import cv2
import numpy as np

class Model(BaseModel):
	def __init__(self):
		super(Model, self).__init__()
		self.name = 'Gaussian Blur'

		self.kernel_size = 5

		# Define options
		self.options = ('kernel_size',)

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
		image = self.linear_to_srgb(image)
		image = (image * 255).astype(np.uint8)
		kernel = self.kernel_size * 2 + 1
		blur = cv2.GaussianBlur(image, (kernel, kernel), 0)
		blur = blur.astype(np.float32) / 255.
		blur = self.srgb_to_linear(blur)

		return [blur]