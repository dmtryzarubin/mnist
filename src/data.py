import typing as ty
from functools import partial

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader


class LitMnist(pl.LightningDataModule):
    def __init__(
        self,
        data_folder: str,
        transforms: ty.Dict[str, torch.nn.Sequential] = None,
        val_pct: float = 0.1,
        batch_size: int = 32,
        random_state: int = 2023,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.transforms = transforms
        self.val_pct = val_pct
        self.batch_size = batch_size
        self.random_state = random_state
        self.num_workers = num_workers
        self._default_loader = partial(
            DataLoader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_data = torchvision.datasets.MNIST(
                self.data_folder, train=True, transform=self.transforms["train"]
            )
            self.train_data, self.val_data = torch.utils.data.random_split(
                train_data,
                [1.0 - self.val_pct, self.val_pct],
                torch.Generator().manual_seed(self.random_state),
            )
        if stage == "test" or stage is None:
            self.test_data = torchvision.datasets.MNIST(
                self.data_folder, train=False, transform=self.transforms["test"]
            )

    def train_dataloader(self):
        return self._default_loader(dataset=self.train_data)

    def val_dataloader(self):
        return self._default_loader(dataset=self.val_data)

    def test_dataloader(self):
        return self._default_loader(dataset=self.test_data)
