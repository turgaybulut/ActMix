from src.datamodules.base import BaseTabularDataModule, TabularDataset
from src.datamodules.uci_classification import (
    AdultCensusDataModule,
    MagicGammaDataModule,
)
from src.datamodules.uci_regression import PowerPlantDataModule

DATAMODULE_REGISTRY: dict[str, type[BaseTabularDataModule]] = {
    "power_plant": PowerPlantDataModule,
    "adult_census": AdultCensusDataModule,
    "magic_gamma": MagicGammaDataModule,
}

__all__ = [
    "BaseTabularDataModule",
    "TabularDataset",
    "AdultCensusDataModule",
    "MagicGammaDataModule",
    "PowerPlantDataModule",
    "DATAMODULE_REGISTRY",
]
