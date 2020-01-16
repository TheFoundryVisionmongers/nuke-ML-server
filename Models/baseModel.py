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

import numpy as np

class BaseModel(object):
    def __init__(self):
        self.name = 'Base model'
        self.options = ()               # List of attribute names that should get exposed in Nuke
        self.buttons = ()               # List of button names that should get exposed in Nuke
        self.inputs = {'input': 3}      # Define Inputs (name, #channels)
        self.outputs = {'output': 3}    # And Outputs (name, #channels)
        pass

    def inference(self, *inputs):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            inputs: A list of images

        # Return:
            The result of the inference as a list of images
        """
        raise NotImplementedError

    def get_options(self):
        """Get a dictionary of exposed options from the model.
        
        To expose options, self.options has to be filled with attribute names.
        Return a dictionary of option names and values.
        """
        opt = {}
        if hasattr(self, 'options'):
            for option in self.options:
                value = getattr(self, option)
                if isinstance(value, unicode):
                    value = str(value)
                assert type(value) in [bool, int, float, str], \
                    'Broadcasted options need to be one of bool, int, float, str.'
                opt[option] = value
        return opt

    def set_options(self, optionsDict):
        """Set the options of the model.

        # Arguments:
            optionsDict: A dictionary of attribute names and values
        """
        for name, value in optionsDict.items():
            setattr(self, name, value)

    def get_buttons(self):
        """Return the defined buttons of the model.

        To expose buttons in Nuke, self.buttons has to be filled with attribute names.
        """
        btn = {}
        if hasattr(self, 'buttons'):
            for button in self.buttons:
                value = getattr(self, button)
                if isinstance(value, unicode):
                    value = str(value)
                assert type(value) in [bool], 'Broadcasted buttons need to be bool.'
                btn[button] = value
        return btn
	
    def set_buttons(self, buttonsDict):
        """Set the buttons of the model.

        # Arguments:
            buttonsDict: A dictionary of attribute names and values
        """
        for name, value in buttonsDict.items():
            setattr(self, name, value)

    def get_inputs(self):
        """Return the defined inputs of the model."""
        return self.inputs

    def get_outputs(self):
        """Return the defined outputs of the model."""
        return self.outputs

    def get_name(self):
        """Return the name of the model."""
        return self.name