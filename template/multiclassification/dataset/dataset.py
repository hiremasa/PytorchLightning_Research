from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from dataset.transformation import Transforms

class TrainDataset(Dataset):
    def __init__(self, image_dir: str, df, transform: Compose, config: dict):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self._config=config
        self.labels=df.labels

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path =
        img_array = Image.open(img_path)
        if self.transform:
            img = self.transform(img_array)
        else:
            raise ValueError("specify transform")

        label = self.df.iloc[index][""]
        return img, label
