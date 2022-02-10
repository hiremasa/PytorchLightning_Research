import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.dataset import TrainDataset
from dataset.transformation import Transforms


class DomainDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df,  config: dict):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        self._config = config

    def train_dataloader(self) -> DataLoader:
        self.train_dataset = TrainUrineDataset(
            image_dir=self._config.image_dir,
            df=self._train_df,
            transform=Transforms(config=self._config).train_transform,
            config=self._config
        )
        return DataLoader(self.train_dataset, sampler=ImbalancedDatasetSampler(self.train_dataset), **self._config.train_loader)

    def val_dataloader(self) -> DataLoader:
        self.valid_dataset = TrainUrineDataset(
            image_dir=self._config.image_dir,
            df=self._val_df,
            transform=Transforms(config=self._config).test_transform,
            config=self._config
        )
        return DataLoader(self.valid_dataset, **self._config.val_loader)

    def test_dataloader(self) -> DataLoader:
        self.test_dataset = TrainUrineDataset(
            image_dir=self._config.image_dir,
            df=self._test_df,
            transform=Transforms(config=self._config).test_transform,
            config=self._config
        )
        return DataLoader(self.test_dataset, **self._config.val_loader)

