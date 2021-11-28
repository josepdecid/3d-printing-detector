import os
from glob import glob
from typing import Callable, Tuple, List

import torch
from stl import mesh
from torch.utils.data import Dataset


class Piece3DPrintDataset(Dataset):
    def __init__(self, root_path: str, transform: Callable):
        self.__root = root_path
        self.__transform = transform

        self.__data = self.__make_dataset()

        self.class_from_idx = {
            idx: Piece3DPrintDataset.__get_class_name_from_path(class_name)
            for idx, class_name in enumerate(self.__data)
        }

        self.__idx_from_class = {class_name: idx for idx, class_name in self.class_from_idx.items()}

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        class_name = self.__data[idx]
        stl_mesh = mesh.Mesh.from_file(class_name)

        x = self.__transform(stl_mesh)
        y = self.__idx_from_class[Piece3DPrintDataset.__get_class_name_from_path(class_name)]

        return x, y

    def __make_dataset(self) -> List[str]:
        return sorted(glob(os.path.join(self.__root, '*.stl')))

    @staticmethod
    def __get_class_name_from_path(path: str):
        return path.split(os.path.sep)[-1][:-len('.stl')]
