import numpy as np


class RandomNoise(object):
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img
