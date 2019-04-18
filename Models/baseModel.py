# Copyright (c) 2019 The Foundry Visionmongers Ltd. All Rights Reserved.
# This is strictly non-commercial.

import numpy as np

class BaseModel(object):
	def __init__(self):
		self.name = 'Base model'
		self.options = ()  # List of attribute names that should get exposed.
		self.inputs = {'input': 3}  # Define Inputs (name, #channels)
		self.outputs = {'output': 3}  # And Outputs (name, #channels)
		pass

	def inference(self, *inputs):
		"""Does an inference on the model with a set of inputs.

		# Arguments:
			inputs: A list of images.

		# Returns:
			The result of the inference as a list of images.
		"""

		raise NotImplementedError

	def get_options(self):
		"""Gets a dictionary of exposed options from the model. To expose options,
		self.options has to be filled with attribute names.

		# Returns:
			A dictionary of option names and values.
		"""
		
		opt = {}
		if hasattr(self, 'options'):
			for option in self.options:
				value = getattr(self, option)
				if isinstance(value, unicode):
					value = str(value)
				assert type(value) in [bool, int, float, str], 'Broadcasted options need to be one of bool, int, float, str.'
				opt[option] = value
		return opt

	def set_options(self, optionsDict):
		"""Sets the options of the model.

		# Arguments:
			optionsDict: A dictionary of attribute names and values.
		"""
		
		for name, value in optionsDict.items():
			setattr(self, name, value)

	def get_inputs(self):
		"""Returns the defined inputs of the model.
		"""

		return self.inputs

	def get_outputs(self):
		"""Returns the defined outputs of the model.
		"""

		return self.outputs

	def get_name(self):
		"""Returns the name of the model.
		"""

		return self.name

	def srgb_to_linear(self, x):
		"""Transforms the image from linear to sRGB.

		# Arguments:
			x: The image to transform.
		"""

		a = 0.055
		x = np.clip(x, 0, 1)
		np_array = np.where(x < 0.04045,
							x / 12.92,
							pow(((x + a) / (1 + a)), 2.4))

		return np_array

	def linear_to_srgb(self, x):
		"""Transforms the image from sRGB to linear.

		# Arguments:
			x: The image to transform.
		"""

		a = 0.055
		x = np.clip(x, 0, 1)
		np_array = np.where(x <= 0.0031308,
							x * 12.92,
							(1 + a) * pow(x, 1 / 2.4) - a)

		return np_array
