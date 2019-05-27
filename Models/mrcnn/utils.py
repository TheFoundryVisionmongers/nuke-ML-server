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

"""Utility functions for mask rcnn model"""

from detectron.utils.collections import AttrDict
import numpy as np

def dict_equal(d1, d2):
    """Recursively compute if two dictionaries are equals both in keys and values.

    # Arguments:
        d1, d2: The two dictionaries to compare
    
    # Return:
        False if any key or value are different, True otherwise
    """
    for k in d1:
        if k not in d2:
            return False
    for k in d2:
        if type(d2[k]) not in (dict, list, AttrDict, np.ndarray):
            if d2[k] != d1[k]:
                return False
        elif type(d2[k]) == "np.ndarray":
            if any(d2[k] != d1[k]):
                return False
        else: # d2[k] dictionary or list
            if type(d1[k]) != type(d2[k]):
                return False
            else:
                if type(d2[k]) == AttrDict or type(d2[k]) == dict:
                    if(not dict_equal(d1[k], d2[k])):
                        return False
    return True