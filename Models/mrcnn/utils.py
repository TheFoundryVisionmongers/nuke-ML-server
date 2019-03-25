# Utility functions for mask rcnn model

from detectron.utils.collections import AttrDict
import numpy as np

def dict_equal(d1, d2):
    """Recursively compute if two dictionaries are equals both in keys and values

        # Arguments:
            d1, d2: The two dictionaries to compare
        
        # Returns:
            False if any key or value are different, True otherwise
    """

    for k in d1:
        if k not in d2:
            return False
    for k in d2:
        if k not in d1:
            return False
        if type(d2[k]) not in (dict, list, AttrDict, np.ndarray):
            if d2[k] != d1[k]:
                return False
        elif type(d2[k]) == "np.ndarray":
            if any(d2[k] != d1[k]):
                return False
        else: #d2[k] dictionary or list
            if type(d1[k]) != type(d2[k]):
                return False
            else:
                if type(d2[k]) == dict:
                    dict_equal(d1[k], d2[k])
                    continue
    return True