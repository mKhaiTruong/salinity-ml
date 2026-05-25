from core.models import BaseModelStrategy
from core.models.xgboost_strategy import XGBoostStrategy
from core.models.lstm_strategy import LSTMStrategy
from core.models.iTransformer_strategy import iTransformerStrategy

from training import TrainingConfig

STRATEGY_REGISTRY = {
    "xgboost": XGBoostStrategy,
    "lstm": LSTMStrategy,
    "iTransformer": iTransformerStrategy,
}

def get_strategy(config: TrainingConfig, **kwargs) -> BaseModelStrategy:
    name = config.model.model_name

    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported model: '{name}'. Choose from: {list(STRATEGY_REGISTRY)}")
    
    strategy_cls = STRATEGY_REGISTRY[name]
    
    if name == "xgboost":
        return strategy_cls(config.model_config)
    elif name == "lstm":
        return strategy_cls(config.model_config, config.model, **kwargs)
    elif name == "iTransformer":
        return strategy_cls(config.model_config, config.model)