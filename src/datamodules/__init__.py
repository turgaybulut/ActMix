from src.datamodules.uci_classification import (
    AdultCensusDataModule,
    MagicGammaDataModule,
)
from src.datamodules.uci_regression import PowerPlantDataModule

__all__ = [
    "AdultCensusDataModule",
    "MagicGammaDataModule",
    "PowerPlantDataModule",
]
