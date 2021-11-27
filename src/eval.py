import os
from typing import Union, Optional

import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from torchvision.transforms import Compose, ToTensor, Normalize

from datasets.Piece3DPrintDataset import Piece3DPrintDataset

eval_transformations = Compose([
    ToTensor(),
    Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


class Evaluator:
    def __init__(self, checkpoint_path: Optional[str], device: str):
        # TODO: Load proper model instead of an empty one
        self.__model = torch.nn.Module()

        if checkpoint_path is not None:
            Evaluator.__load_checkpoint(self.__model, checkpoint_path)

        # Set the model to the specified device
        self.__device = torch.device(device)
        self.__model.to(self.__device)
        # Set in evaluation model to avoid unexpected random variations
        self.__model.eval()

    def evaluate(self, img: Union[np.ndarray, torch.tensor]):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        # Transform the image with the evaluation transformations
        img = eval_transformations(img)
        # Add a new dimension for the batches with a single instance
        img = torch.unsqueeze(img, 0)
        # Send the instance to evaluate to the same device as the model
        img = img.to(self.__device)

        with torch.no_grad():
            prediction = self.__model(img)
            return prediction

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
    predicted_class_idx = evaluator.evaluate(img)

    dataset = Piece3DPrintDataset(os.environ.get('DATASET_PATH'), eval_transformations)
    class_name = dataset.class_from_idx[predicted_class_idx]

    print(class_name)


if __name__ == '__main__':
    main()
