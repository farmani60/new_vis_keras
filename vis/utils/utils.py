import os
import tempfile
import math
import json
import six

import numpy as np
import matplotlib.font_manager as fontman

from skimage import io, transform
from keras import backend as K
from keras.models import load_model

import logging
logger = logging.getLogger(__name__)

try:
    import PIL as pil
    from PIL import ImageFont
    from PIL import Image
    from PIL import ImageDraw
except ImportError:
    pil = None

# Clobals
_CLASS_INDEX = None


def _check_pil():
    if not pil:
        raise ImportError("Failed to import PIL, You must install Pillow.")


def reverse_enumerate(iterable):
    """Enumerate over an iterable in reverse order while retaining proper indexes, without creating any copies.
    """
    return zip(reversed(range(len(iterable))), reversed(iterable))


def listify(value):
    """Ensures that the value is a list. If it is not a list, it creates a new list with `value` as an item.
    """
    if not isinstance(value, list):
        value = [value]
    return value


class _BackendAgnosticImageSlice:
    """Utility class to make image slicing uniform across various `image_data_format`.
    """
    def __getitem__(self, item_slice):
        """Assuming a slice for shape `(samples, channels, image_dims...)`
        """
        if K.image_data_format() == "channels_first":
            return item_slice
        else:
            # Move channel index to last position.
            item_slice = list(item_slice)
            item_slice.append(item_slice.pop(1))
            return tuple(item_slice)


"""Slice utility to make image slicing uniform across various `image_data_format`.
Example:
    conv_layer[utils.slicer[:, filter_idx, :, :]] will work for both `channels_first` and `channels_last` image
    data formats even though, in tensorflow, slice should be conv_layer[utils.slicer[:, :, :, filter_idx]]
"""
slicer = _BackendAgnosticImageSlice()

