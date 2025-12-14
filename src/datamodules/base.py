from abc import ABC, abstractmethod
from typing import Literal

import lightning as L
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=target_dtype)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class BaseTabularDataModule(L.LightningDataModule, ABC):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.10,
        test_split: float = 0.10,
        seed: int = 1192,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.feature_scaler = StandardScaler()

        self.train_dataset: TabularDataset | None = None
        self.val_dataset: TabularDataset | None = None
        self.test_dataset: TabularDataset | None = None

        self._input_dim: int | None = None
        self._output_dim: int | None = None

    @property
    @abstractmethod
    def task(self) -> Literal["classification", "regression"]:
        pass

    @property
    def input_dim(self) -> int:
        assert self._input_dim is not None
        return self._input_dim

    @property
    def output_dim(self) -> int:
        assert self._output_dim is not None
        return self._output_dim

    @property
    def target_dtype(self) -> torch.dtype:
        return torch.long if self.task == "classification" else torch.float32

    def _create_dataloader(self, dataset: TabularDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self._create_dataloader(self.test_dataset, shuffle=False)
