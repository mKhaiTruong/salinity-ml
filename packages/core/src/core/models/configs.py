from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    model_name:  str
    window_size: int
    horizon:     int

@dataclass(frozen=True) 
class XGBoostConfig:
    n_estimators:  int
    max_depth:     int
    learning_rate: float
    subsample:     float

@dataclass(frozen=True)
class LSTMConfig:
    hidden_size:   int
    num_layers:    int
    dropout:       float
    
    epochs:        int
    learning_rate: float
    batch_size:    int
    patience:      int
    
@dataclass(frozen=True)
class iTransformerConfig:
    hidden_size:   int
    epochs:        int
    learning_rate: float
    batch_size:    int
    patience:      int