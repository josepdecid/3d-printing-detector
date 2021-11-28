import math
import os
from typing import Optional

import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from torchvision.transforms import ToTensor, Compose, Normalize

from datasets.Piece3DPrintDataset import Piece3DPrintDataset


class Evaluator:
    def __init__(self, checkpoint_path: Optional[str], device: str):
        # TODO: Load proper model instead of an empty one
        self.__model = torch.nn.Module()
        if checkpoint_path is not None:
            self.__model = Evaluator.__load_checkpoint(self.__model, checkpoint_path)

        # Simple eval transformations with only tensorized and normalized data
        self.__eval_transformations = Compose([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.__dataset = Piece3DPrintDataset(os.environ.get('DATASET_PATH'), self.__eval_transformations)

        # Set the model to the specified device
        self.__device = torch.device(device)
        self.__model.to(self.__device)
        # Set in evaluation model to avoid unexpected random variations
        self.__model.eval()

    def evaluate(self, img: np.ndarray) -> str:
        # Transform the image with the evaluation transformations
        img = self.__eval_transformations(img)
        # Add a new dimension for the batches with a single instance
        img = torch.unsqueeze(img, 0)
        # Send the instance to evaluate to the same device as the model
        img = img.to(self.__device)

        with torch.no_grad():
            try:
                predicted_class_idx = self.__model(img)
            # TODO: Remove when the model is updated
            except NotImplementedError:
                predicted_class_idx = math.floor(8 * np.random.rand())

        return self.__dataset.class_from_idx[predicted_class_idx]

    @staticmethod
    def __load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
        model.load_state_dict(torch.load(checkpoint_path))
        return model


@hydra.main(config_path=os.path.join('..', 'configs'), config_name='eval')
def main(cfg: DictConfig) -> None:
    checkpoints_folder = os.environ.get('CHECKPOINTS_PATH')
    checkpoint_path = os.path.join(checkpoints_folder, cfg.checkpoint)

    img = np.array(Image.open(cfg.image))

    evaluator = Evaluator(checkpoint_path, cfg.device)
    class_name = evaluator.evaluate(img)

    print(class_name)


if __name__ == '__main__':
    main()
