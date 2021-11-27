import os
from glob import glob
from logging import warning
from typing import List

import cv2
import numpy as np
from PIL import Image

VALID_BACKGROUND_EXTENSIONS = ['png', 'jpg', 'jpeg']


class RandomBackground(object):
    """
    With a probability `background_img_prob`, it selects a random image to use as a background from the `folder_path`.
    Otherwise, it creates a random single-color image by sampling uniformly the three color components.

    The selected image is cropped by a random factor between `min_cropping_factor` and 1.
    It extracts the piece from an image with an alpha channel, separating the foreground and the background.
    Then, it merges piece and background according to the alpha channels of the original image.
    The returned image loses the alpha channel, becoming a RGB image.
    """

    def __init__(self, folder_path: str, min_cropping_factor: int = 0.25, background_img_prob: int = 0.9):
        background_files = glob(os.path.join(folder_path, '*'))
        assert os.path.exists(folder_path), 'The specified background folder does not exist.'
        assert len(background_files), 'There are no available backgrounds in the specified folder.'
        self.__backgrounds = RandomBackground.__filter_supported_extensions(background_files)

        assert 0 <= min_cropping_factor <= 1, 'The background image probability must be in range [0, 1].'
        self.__min_cropping_percentage = min_cropping_factor

        assert 0 <= background_img_prob <= 1, 'The cropping percentage must be in range [0, 1].'
        self.__background_img_prob = background_img_prob

    def __call__(self, img, annotations):
        if img.shape[2] == 3:
            warning('Image does not have an alpha channel, skipping this step.')
            return img[:, :, :3], annotations

        # Select a random background
        if np.random.random() < self.__background_img_prob:
            random_idx = np.random.randint(0, len(self.__backgrounds))
            random_bg = np.array(Image.open(self.__backgrounds[random_idx]))
        else:
            r_component = np.ones(img.shape[:2]) * np.random.randint(0, 255)
            g_component = np.ones(img.shape[:2]) * np.random.randint(0, 255)
            b_component = np.ones(img.shape[:2]) * np.random.randint(0, 255)
            random_bg = np.stack([r_component, g_component, b_component], axis=2)

        # Crop the background randomly
        cropping_percentage = np.random.uniform(low=self.__min_cropping_percentage, high=1.0)
        cropped_width = int(random_bg.shape[0] * cropping_percentage)
        cropped_height = int(random_bg.shape[1] * cropping_percentage)
        min_x = np.random.randint(0, random_bg.shape[0] - cropped_width)
        min_y = np.random.randint(0, random_bg.shape[1] - cropped_height)
        random_bg = random_bg[min_x:min_x + cropped_width, min_y:min_y + cropped_height, :]

        # Resize image to the same size of the original image
        random_bg = cv2.resize(random_bg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Merge gauge and background according to the alpha channels of the original image
        img = np.where((img[:, :, 3] > 100)[..., None], img[:, :, :3], random_bg)

        return img, annotations

    @staticmethod
    def __filter_supported_extensions(files: List[str]) -> List[str]:
        return list(filter(lambda x: x.split('.')[-1] in VALID_BACKGROUND_EXTENSIONS, files))
