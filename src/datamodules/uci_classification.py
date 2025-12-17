from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo

from src.datamodules.base import BaseTabularDataModule, TabularDataset


class AdultCensusDataModule(BaseTabularDataModule):
    UCI_DATASET_ID = 2

    CATEGORICAL_COLUMNS = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    CONTINUOUS_COLUMNS = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

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
        self.one_hot_encoder: OneHotEncoder | None = None
        self._output_dim = 2

    @property
    def task(self) -> Literal["classification"]:
        return "classification"

    def prepare_data(self) -> None:
        fetch_ucirepo(id=self.UCI_DATASET_ID)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.replace("?", np.nan).dropna()

    def _encode_target(self, targets: pd.DataFrame) -> np.ndarray:
        target_series = targets.iloc[:, 0].astype(str).str.strip().str.rstrip(".")
        return (target_series == ">50K").astype(np.int64).values

    def _process_features(
        self,
        features_df: pd.DataFrame,
        fit: bool = False,
    ) -> np.ndarray:
        available_continuous = [
            col for col in self.CONTINUOUS_COLUMNS if col in features_df.columns
        ]
        available_categorical = [
            col for col in self.CATEGORICAL_COLUMNS if col in features_df.columns
        ]

        continuous_features = features_df[available_continuous].values.astype(
            np.float32
        )

        if available_categorical:
            categorical_values = features_df[available_categorical].apply(
                lambda col: col.astype(str).str.strip()
            )
            if fit:
                self.one_hot_encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                categorical_features = self.one_hot_encoder.fit_transform(
                    categorical_values
                )
            else:
                assert self.one_hot_encoder is not None
                categorical_features = self.one_hot_encoder.transform(
                    categorical_values
                )
            categorical_features = categorical_features.astype(np.float32)
        else:
            categorical_features = np.empty((len(features_df), 0), dtype=np.float32)

        return np.hstack([continuous_features, categorical_features])

    def _scale_continuous_features(
        self,
        X: np.ndarray,
        features_df: pd.DataFrame,
        fit: bool = False,
    ) -> np.ndarray:
        continuous_dim = len(
            [col for col in self.CONTINUOUS_COLUMNS if col in features_df.columns]
        )
        if fit:
            self.feature_scaler.fit(X[:, :continuous_dim])
        X[:, :continuous_dim] = self.feature_scaler.transform(X[:, :continuous_dim])
        return X

    def setup(self, stage: str | None = None) -> None:
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)

        features_df = dataset.data.features
        targets_df = dataset.data.targets

        combined_df = pd.concat([features_df, targets_df], axis=1)
        combined_df = self._handle_missing_values(combined_df)

        features_df = combined_df[features_df.columns]
        targets_df = combined_df[targets_df.columns]

        y = self._encode_target(targets_df)

        no_test_split = self.test_split <= 0.0
        no_val_split = self.val_split <= 0.0

        if no_test_split and no_val_split:
            X = self._process_features(features_df, fit=True)
            X = self._scale_continuous_features(X, features_df, fit=True)
            self._input_dim = X.shape[1]
            self.train_dataset = TabularDataset(X, y, self.target_dtype)
            self.val_dataset = self.train_dataset
            self.test_dataset = self.train_dataset
            return

        if no_test_split:
            train_idx, val_idx = train_test_split(
                np.arange(len(features_df)),
                test_size=self.val_split,
                random_state=self.seed,
                stratify=y,
            )
            X_train = self._process_features(features_df.iloc[train_idx], fit=True)
            X_val = self._process_features(features_df.iloc[val_idx], fit=False)
            X_train = self._scale_continuous_features(X_train, features_df, fit=True)
            X_val = self._scale_continuous_features(X_val, features_df, fit=False)
            self._input_dim = X_train.shape[1]
            self.train_dataset = TabularDataset(
                X_train, y[train_idx], self.target_dtype
            )
            self.val_dataset = TabularDataset(X_val, y[val_idx], self.target_dtype)
            self.test_dataset = self.val_dataset
            return

        train_val_idx, test_idx = train_test_split(
            np.arange(len(features_df)),
            test_size=self.test_split,
            random_state=self.seed,
            stratify=y,
        )

        if no_val_split:
            X_train = self._process_features(features_df.iloc[train_val_idx], fit=True)
            X_test = self._process_features(features_df.iloc[test_idx], fit=False)
            X_train = self._scale_continuous_features(X_train, features_df, fit=True)
            X_test = self._scale_continuous_features(X_test, features_df, fit=False)
            self._input_dim = X_train.shape[1]
            self.train_dataset = TabularDataset(
                X_train, y[train_val_idx], self.target_dtype
            )
            self.val_dataset = self.train_dataset
            self.test_dataset = TabularDataset(X_test, y[test_idx], self.target_dtype)
            return

        val_ratio = self.val_split / (1 - self.test_split)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=y[train_val_idx],
        )

        X_train = self._process_features(features_df.iloc[train_idx], fit=True)
        X_val = self._process_features(features_df.iloc[val_idx], fit=False)
        X_test = self._process_features(features_df.iloc[test_idx], fit=False)

        X_train = self._scale_continuous_features(X_train, features_df, fit=True)
        X_val = self._scale_continuous_features(X_val, features_df, fit=False)
        X_test = self._scale_continuous_features(X_test, features_df, fit=False)

        self._input_dim = X_train.shape[1]

        self.train_dataset = TabularDataset(X_train, y[train_idx], self.target_dtype)
        self.val_dataset = TabularDataset(X_val, y[val_idx], self.target_dtype)
        self.test_dataset = TabularDataset(X_test, y[test_idx], self.target_dtype)


class MagicGammaDataModule(BaseTabularDataModule):
    UCI_DATASET_ID = 159

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
        self.label_encoder = LabelEncoder()
        self._input_dim = 10
        self._output_dim = 2

    @property
    def task(self) -> Literal["classification"]:
        return "classification"

    def prepare_data(self) -> None:
        fetch_ucirepo(id=self.UCI_DATASET_ID)

    def setup(self, stage: str | None = None) -> None:
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)

        X = dataset.data.features.values.astype(np.float32)
        y = self.label_encoder.fit_transform(dataset.data.targets.values.ravel())

        no_test_split = self.test_split <= 0.0
        no_val_split = self.val_split <= 0.0

        if no_test_split and no_val_split:
            X_scaled = self.feature_scaler.fit_transform(X)
            self.train_dataset = TabularDataset(X_scaled, y, self.target_dtype)
            self.val_dataset = self.train_dataset
            self.test_dataset = self.train_dataset
            return

        if no_test_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_split, random_state=self.seed, stratify=y
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
            X, y, test_size=self.test_split, random_state=self.seed, stratify=y
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
            X_train_val,
            y_train_val,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=y_train_val,
        )

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        self.train_dataset = TabularDataset(X_train_scaled, y_train, self.target_dtype)
        self.val_dataset = TabularDataset(X_val_scaled, y_val, self.target_dtype)
        self.test_dataset = TabularDataset(X_test_scaled, y_test, self.target_dtype)
