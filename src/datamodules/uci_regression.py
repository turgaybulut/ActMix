import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from ucimlrepo import fetch_ucirepo


class TabularRegressionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class PowerPlantDataModule(L.LightningDataModule):
    UCI_DATASET_ID = 294

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

        self.train_dataset: TabularRegressionDataset | None = None
        self.val_dataset: TabularRegressionDataset | None = None
        self.test_dataset: TabularRegressionDataset | None = None

        self.input_dim = 4
        self.output_dim = 1

    def prepare_data(self) -> None:
        fetch_ucirepo(id=self.UCI_DATASET_ID)

    def setup(self, stage: str | None = None) -> None:
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)

        X = dataset.data.features.values.astype(np.float32)
        y = dataset.data.targets.values.astype(np.float32).reshape(-1, 1)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.seed
        )

        val_ratio = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=self.seed
        )

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        self.train_dataset = TabularRegressionDataset(X_train_scaled, y_train)
        self.val_dataset = TabularRegressionDataset(X_val_scaled, y_val)
        self.test_dataset = TabularRegressionDataset(X_test_scaled, y_test)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
