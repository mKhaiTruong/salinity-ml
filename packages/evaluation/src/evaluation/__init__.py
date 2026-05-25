from dataclasses import dataclass
from pathlib import Path

from core.models.configs import (
    ModelConfig, XGBoostConfig, LSTMConfig, iTransformerConfig
)

@dataclass(frozen=True)
class EvaluationModelConfig:
    model_name:  str    
    window_size: int
    horizon:     int
    model_dir:   Path
    onnx_dir:    Path

ModelSpecificConfig = XGBoostConfig | LSTMConfig | iTransformerConfig

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir:       Path
    data_dir:       Path
    encoders_dir:   Path
    
    model:        EvaluationModelConfig
    model_config: ModelSpecificConfig