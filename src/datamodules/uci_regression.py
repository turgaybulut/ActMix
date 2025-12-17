from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from src.datamodules.base import BaseTabularDataModule, TabularDataset


class PowerPlantDataModule(BaseTabularDataModule):
    UCI_DATASET_ID = 294

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.10,
        test_split: float = 0.10,
        seed: int = 1192,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            pin_memory=pin_memory,
        )
        self._input_dim = 4
        self._output_dim = 1

    @property
    def task(self) -> Literal["regression"]:
        return "regression"

    def prepare_data(self) -> None:
        fetch_ucirepo(id=self.UCI_DATASET_ID)

    def setup(self, stage: str | None = None) -> None:
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)

        X = dataset.data.features.values.astype(np.float32)
        y = dataset.data.targets.values.astype(np.float32).reshape(-1, 1)

        no_test_split = self.test_split <= 0.0
        no_val_split = self.val_split <= 0.0

        if no_test_split and no_val_split:
            X_train_scaled = self.feature_scaler.fit_transform(X)
            self.train_dataset = TabularDataset(X_train_scaled, y, self.target_dtype)
            self.val_dataset = self.train_dataset
            self.test_dataset = self.train_dataset
            return

        if no_test_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_split, random_state=self.seed
            )
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            self.train_dataset = TabularDataset(
                X_train_scaled, y_train, self.target_dtype
            )
            self.val_dataset = TabularDataset(X_val_scaled, y_val, self.target_dtype)
            self.test_dataset = self.val_dataset
            return

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.seed
        )

        if no_val_split:
            X_train_scaled = self.feature_scaler.fit_transform(X_train_val)
            X_test_scaled = self.feature_scaler.transform(X_test)
            self.train_dataset = TabularDataset(
                X_train_scaled, y_train_val, self.target_dtype
            )
            self.val_dataset = self.train_dataset
            self.test_dataset = TabularDataset(X_test_scaled, y_test, self.target_dtype)
            return

        val_ratio = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=self.seed
        )

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        self.train_dataset = TabularDataset(X_train_scaled, y_train, self.target_dtype)
        self.val_dataset = TabularDataset(X_val_scaled, y_val, self.target_dtype)
        self.test_dataset = TabularDataset(X_test_scaled, y_test, self.target_dtype)
