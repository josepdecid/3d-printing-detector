import hashlib
import io
import os
import shutil
import time
from glob import glob
from typing import List

import cv2
import numpy as np
import requests
from PIL import Image
from selenium import webdriver


class RandomBackground(object):
    """
    With a probability `background_img_prob`, it selects a random image to use as a background from the `folder_path`.
    Otherwise, it creates a random single-color image by sampling uniformly the three color components.

    The selected image is cropped by a random factor between `min_cropping_factor` and 1.
    It extracts the piece from an image with an alpha channel, separating the foreground and the background.
    Then, it merges piece and background according to the alpha channels of the original image.
    The returned image loses the alpha channel, becoming a RGB image.
    """

    def __init__(self,
                 search_terms: List[str], images_per_term: int = 20,
                 backgrounds_folder: str = '.backgrounds', download_bg: bool = False,
                 min_cropping_factor: int = 0.25, background_img_prob: int = 0.9):

        self.__fetch_background_images(backgrounds_folder, download_bg, search_terms, images_per_term)
        self.__background_files = glob(os.path.join(backgrounds_folder, '*', '*'))

        assert 0 <= min_cropping_factor <= 1, 'The background image probability must be in range [0, 1].'
        self.__min_cropping_percentage = min_cropping_factor

        assert 0 <= background_img_prob <= 1, 'The cropping percentage must be in range [0, 1].'
        self.__background_img_prob = background_img_prob

    def __call__(self, img):
        # Select a random background
        if np.random.random() < self.__background_img_prob:
            random_idx = np.random.randint(0, len(self.__background_files))
            random_bg = np.array(Image.open(self.__background_files[random_idx]))
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
        img = np.where((img[:, :, 2] != 255)[..., None], img[:, :, :3], random_bg)

        return img

    @staticmethod
    def __fetch_background_images(backgrounds_folder: str, download_bg: bool,
                                  search_terms: List[str], images_per_term: int):
        if os.path.exists(backgrounds_folder) and not download_bg:
            return

        if os.path.exists(backgrounds_folder):
            print('Removing existing background images...')
            shutil.rmtree(backgrounds_folder)
        os.mkdir(backgrounds_folder)

        print('Downloading background images...')
        for term in search_terms:
            target_folder = os.path.join(backgrounds_folder, '_'.join(term.lower().split(' ')))

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            with webdriver.Chrome(executable_path=os.environ.get('SELENIUM_DRIVER_PATH')) as wd:
                res = RandomBackground.fetch_image_urls(term, images_per_term, wd=wd, sleep_between_interactions=0.5)
                for elem in res:
                    RandomBackground.__persist_image(target_folder, elem)

    @staticmethod
    def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: float = 1):
        def scroll_to_end():
            wd.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(sleep_between_interactions)

        # build the google query
        search_url = 'https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img'

        # load the page
        wd.get(search_url.format(q=query))

        image_urls = set()
        image_count = 0
        results_start = 0
        while image_count < max_links_to_fetch:
            scroll_to_end()

            # get all image thumbnail results
            thumbnail_results = wd.find_elements_by_css_selector('img.Q4LuWd')
            number_results = len(thumbnail_results)

            print(f'Found: {number_results} search results. Extracting links from {results_start}:{number_results}')

            for img in thumbnail_results[results_start:number_results]:
                # Try to click every thumbnail such that we can get the real image behind it
                img.click()
                time.sleep(sleep_between_interactions)

                # Extract image urls
                actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
                for actual_image in actual_images:
                    if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                        image_urls.add(actual_image.get_attribute('src'))

                image_count = len(image_urls)

                if len(image_urls) >= max_links_to_fetch:
                    print(f'Found: {len(image_urls)} image links, done!')
                    break
            else:
                print(f'Found: {len(image_urls)} image links, looking for more ...')
                time.sleep(30)
                return

            results_start = len(thumbnail_results)

        return image_urls

    @staticmethod
    def __persist_image(folder_path: str, url: str):
        try:
            image_content = requests.get(url).content
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
            with open(file_path, 'wb') as f:
                image.save(f, 'JPEG', quality=85)
            print(f'SUCCESS - saved {url} - as {file_path}')

        except Exception as e:
            print(f'ERROR - Could not download or save {url} - {e}')
