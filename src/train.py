import os

import hydra
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from datasets.Piece3DPrintDataset import Piece3DPrintDataset
from transforms.random_background import RandomBackground
from transforms.random_projection import RandomProjection


@hydra.main(config_path=os.path.join('..', 'configs'), config_name='train')
def main(cfg: DictConfig) -> None:
    load_dotenv()

    dataset = Piece3DPrintDataset(os.environ.get('DATASET_PATH'), Compose([
        # Project 3D model into a random-oriented 2D image
        RandomProjection(
            azimuth=cfg.illumination.azimuth,
            altitude=cfg.illumination.altitude,
            darkest_shadow_surface=cfg.illumination.darkest_shadow_surface,
            brightest_lit_surface=cfg.illumination.brightest_lit_surface,
        ),

        # Add a randomized background with different cropping sizes
        RandomBackground(
            search_terms=cfg.backgrounds.search_terms,
            download_bg=cfg.backgrounds.download,
            images_per_term=cfg.backgrounds.images_per_term
        ),

        ToTensor()
    ]))

    dataloader = DataLoader(dataset, 1)

    for img in dataloader:
        plt.imshow(img[0, :, :, :].permute(1, 2, 0).numpy())
        plt.show()


if __name__ == '__main__':
    main()
