from dataclasses import dataclass
from pathlib import Path

from core.models.configs import (
    ModelConfig,
    XGBoostConfig, LSTMConfig, iTransformerConfig
)

@dataclass(frozen=True)
class DataConfig:
    train_data_dir: Path
    valid_data_dir: Path

ModelSpecificConfig = XGBoostConfig | LSTMConfig | iTransformerConfig

@dataclass(frozen=True)
class TrainingConfig:
    root_dir:       Path
    model_dir:      Path
    encoders_dir:   Path
    
    model:  ModelConfig
    data:   DataConfig
    
    model_config:   ModelSpecificConfig