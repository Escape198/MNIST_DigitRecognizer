import numpy as np


def normalize_data(data):
    return data / 255.0


def reshape_data(data):
    return np.expand_dims(data, axis=-1)
