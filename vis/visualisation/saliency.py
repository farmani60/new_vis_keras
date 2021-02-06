import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool1D, MaxPool2D, MaxPool3D
from keras.layers.wrappers import Wrapper
from keras import backend as K
from keras.models import load_model



def _find_penultimate_layer(model, layer_idx, penultimate_layer_idx):
    """Searches for the nearest penultimate `Conv` or `Pooling` layer.

    Args:
        model: The `keras.models.Model` instance.
        layer_idx: The layer index within `model.layers`.
        penultimate_layer_idx: The pre-layer to `layer_idx`. If set to None, the nearest penultimate
            `Conv` or `Pooling` layer is used.
    Returns:
        The penultimate layer.
    """


if __name__ == "__main__":
    model_dir = "/home/reza/Documents/cnn_v0.h5"
    model = load_model(model_dir)
    print(model.summary())