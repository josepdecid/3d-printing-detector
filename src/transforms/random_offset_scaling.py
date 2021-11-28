from typing import Tuple

import cv2
import numpy as np


class RandomOffsetScalingAndPadding(object):
    def __init__(self, target_size: Tuple[int, int]):
        self.__target_width, self.__target_height = target_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_background = (255 * np.ones(shape=(self.__target_height, self.__target_width, 3))).astype(np.uint8)

        new_w = np.random.randint(self.__target_width / 2, self.__target_width)
        new_h = np.random.randint(self.__target_height / 2, self.__target_height)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        random_x = np.random.randint(0, self.__target_width - new_w)
        random_y = np.random.randint(0, self.__target_height - new_h)

        img_background[random_y:random_y + new_h, random_x:random_x + new_w, :] = img_resized
        return img_background
