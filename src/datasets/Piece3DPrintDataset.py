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

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        stl_mesh = mesh.Mesh.from_file(self.__data[idx])
        return self.__transform(stl_mesh)

    def __make_dataset(self) -> List[str]:
        return sorted(glob(os.path.join(self.__root, '*.stl')))
