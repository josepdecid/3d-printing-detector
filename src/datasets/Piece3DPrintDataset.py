import os
from glob import glob
from typing import Callable, Tuple, List

import torch
from PIL import Image
from stl import mesh
from torch.utils.data import Dataset


class Piece3DPrintDataset(Dataset):
    def __init__(self, root_path: str, transform: Callable, eval_mode=False):
        self.__root = root_path
        self.__transform = transform

        self.__eval_mode = eval_mode
        self.__data = self.__make_dataset()

        get_class_method = Piece3DPrintDataset.__get_class_name_from_eval_path \
            if eval_mode else Piece3DPrintDataset.__get_class_name_from_path

        self.class_from_idx = {}
        i = 0
        for data in self.__data:
            if get_class_method(data) not in self.class_from_idx.values():
                self.class_from_idx[i] = get_class_method(data)
                i += 1

        self.__idx_from_class = {class_name: idx for idx, class_name in self.class_from_idx.items()}

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        class_path = self.__data[idx]

        if self.__eval_mode:
            class_name = Piece3DPrintDataset.__get_class_name_from_eval_path(class_path)
            stl_mesh = Image.open(class_path)
        else:
            class_name = Piece3DPrintDataset.__get_class_name_from_path(class_path)
            stl_mesh = mesh.Mesh.from_file(class_path)

        x = self.__transform(stl_mesh)
        y = self.__idx_from_class[class_name]

        return x, y

    def __make_dataset(self) -> List[str]:
        if not self.__eval_mode:
            return sorted(glob(os.path.join(self.__root, '*.stl')))
        else:
            return sorted(glob(os.path.join(self.__root, '*.jpeg')))

    @staticmethod
    def __get_class_name_from_path(path: str) -> str:
        return path.split(os.path.sep)[-1][:-len('.stl')]

    @staticmethod
    def __get_class_name_from_eval_path(path: str) -> str:
        file_name = path.split(os.path.sep)[-1][:-len('.jpeg')]
        return file_name.split('@')[0]
