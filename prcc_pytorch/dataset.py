from dataclasses import dataclass
from glob import glob
from os.path import join
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


@dataclass
class Person:
    id: int
    cam: str
    path: str


class PRCCDataset(Dataset):
    def __init__(self, root: str, rgb_sketch: str, train_val_test: str, transforms: Compose = None):
        if rgb_sketch not in ("rgb", "sketch"):
            raise ValueError
        if train_val_test not in ("train", "val", "test"):
            raise ValueError

        self._data = self.read_data(root, rgb_sketch, train_val_test)
        self._transforms = transforms

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        p = self._data[idx]

        image = Image.open(p.path)
        if self._transforms is not None:
            image = self._transforms(image)

        return image, p.id

    @property
    def data(self):
        return self._data

    @staticmethod
    def read_data(root: str, rgb_sketch: str, train_val_test: str) -> List[Person]:
        data: List[Person] = []

        # directory structure is different for train or val and test
        if train_val_test in ("train", "val"):
            for path in sorted(glob(join(root, rgb_sketch, train_val_test, "**/*.jpg"), recursive=True)):
                data.append(Person(id=int(path.split("/")[-2]), cam=path.split("/")[-1][0], path=path))
        else:
            for path in sorted(glob(join(root, rgb_sketch, train_val_test, "**/*.jpg"), recursive=True)):
                data.append(Person(id=int(path.split("/")[-2]), cam=path.split("/")[-3], path=path))

        return data
