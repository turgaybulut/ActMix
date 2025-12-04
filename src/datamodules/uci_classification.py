import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from ucimlrepo import fetch_ucirepo


class TabularClassificationDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class AdultCensusDataModule(L.LightningDataModule):
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
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.feature_scaler = StandardScaler()
        self.one_hot_encoder: OneHotEncoder | None = None

        self.train_dataset: TabularClassificationDataset | None = None
        self.val_dataset: TabularClassificationDataset | None = None
        self.test_dataset: TabularClassificationDataset | None = None

        self.input_dim: int | None = None
        self.output_dim = 2

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
                    handle_unknown="ignore", sparse=False
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

    def setup(self, stage: str | None = None) -> None:
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)

        features_df = dataset.data.features
        targets_df = dataset.data.targets

        combined_df = pd.concat([features_df, targets_df], axis=1)
        combined_df = self._handle_missing_values(combined_df)

        features_df = combined_df[features_df.columns]
        targets_df = combined_df[targets_df.columns]

        y = self._encode_target(targets_df)

        train_val_idx, test_idx = train_test_split(
            np.arange(len(features_df)),
            test_size=self.test_split,
            random_state=self.seed,
            stratify=y,
        )

        val_ratio = self.val_split / (1 - self.test_split)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=y[train_val_idx],
        )

        train_features_df = features_df.iloc[train_idx]
        val_features_df = features_df.iloc[val_idx]
        test_features_df = features_df.iloc[test_idx]

        X_train = self._process_features(train_features_df, fit=True)
        X_val = self._process_features(val_features_df, fit=False)
        X_test = self._process_features(test_features_df, fit=False)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        continuous_dim = len(
            [col for col in self.CONTINUOUS_COLUMNS if col in features_df.columns]
        )

        self.feature_scaler.fit(X_train[:, :continuous_dim])
        X_train[:, :continuous_dim] = self.feature_scaler.transform(
            X_train[:, :continuous_dim]
        )
        X_val[:, :continuous_dim] = self.feature_scaler.transform(
            X_val[:, :continuous_dim]
        )
        X_test[:, :continuous_dim] = self.feature_scaler.transform(
            X_test[:, :continuous_dim]
        )

        self.input_dim = X_train.shape[1]

        self.train_dataset = TabularClassificationDataset(X_train, y_train)
        self.val_dataset = TabularClassificationDataset(X_val, y_val)
        self.test_dataset = TabularClassificationDataset(X_test, y_test)

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


class MagicGammaDataModule(L.LightningDataModule):
    UCI_DATASET_ID = 159

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
        self.label_encoder = LabelEncoder()

        self.train_dataset: TabularClassificationDataset | None = None
        self.val_dataset: TabularClassificationDataset | None = None
        self.test_dataset: TabularClassificationDataset | None = None

        self.input_dim = 10
        self.output_dim = 2

    def prepare_data(self) -> None:
        fetch_ucirepo(id=self.UCI_DATASET_ID)

    def setup(self, stage: str | None = None) -> None:
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)

        X = dataset.data.features.values.astype(np.float32)
        y = self.label_encoder.fit_transform(dataset.data.targets.values.ravel())

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.seed, stratify=y
        )

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

        self.train_dataset = TabularClassificationDataset(X_train_scaled, y_train)
        self.val_dataset = TabularClassificationDataset(X_val_scaled, y_val)
        self.test_dataset = TabularClassificationDataset(X_test_scaled, y_test)

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
