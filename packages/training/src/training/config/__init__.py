import os
from core.constants import *
from core import read_yaml, create_directories

from core.models.configs import (
    ModelConfig,
    XGBoostConfig, LSTMConfig, iTransformerConfig
)

from training import (
    ModelSpecificConfig, DataConfig, TrainingConfig
)

class ConfigurationManager:
    
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    _CONFIG_BUILDERS = {
        'xgboost': '_get_xgboost_config',
        'lstm':    '_get_lstm_config',
        'iTransformer': '_get_itransformer_config',
    }
    
    def _get_model_config(self, model_name: str) -> ModelSpecificConfig:
        builder_name = self._CONFIG_BUILDERS.get(model_name)
        if not builder_name:
            raise ValueError(
                f"Unsupported model: '{model_name}'. "
                f"Choose from: {list(self._CONFIG_BUILDERS)}"
            )
        return getattr(self, builder_name)()
    
    def _get_xgboost_config(self) -> XGBoostConfig:
        p = self.params.xgboost
        return XGBoostConfig(
            n_estimators   = p.n_estimators,
            max_depth      = p.max_depth,
            learning_rate  = p.learning_rate,
            subsample      = p.subsample
        )
    
    def _get_lstm_config(self) -> LSTMConfig:
        p = self.params.lstm
        return LSTMConfig(
            hidden_size    = p.hidden_size,
            num_layers     = p.num_layers,
            dropout        = p.dropout,
            
            epochs         = p.epochs,
            batch_size     = p.batch_size,
            learning_rate  = p.learning_rate,
            patience       = p.patience,
        )
    
    def _get_itransformer_config(self) -> iTransformerConfig:
        p = self.params.iTransformer
        return iTransformerConfig(
            hidden_size    = p.hidden_size,
            epochs         = p.epochs,
            batch_size     = p.batch_size,
            learning_rate  = p.learning_rate,
            patience       = p.patience,
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training_config
        params = self.params.training_params
        
        model_dir = Path(config.root_dir) / params.model_name

        create_directories([
            config.root_dir,
            config.encoders_dir,
            model_dir,
        ])
        
        return TrainingConfig(
            root_dir        = Path(config.root_dir),
            model_dir       = model_dir, 
            encoders_dir    = Path(config.encoders_dir),
            
            model = ModelConfig(
                model_name      = params.model_name,
                window_size     = params.sliding_window_size,
                horizon         = params.sliding_horizon
            ),
            data = DataConfig(
                train_data_dir  = Path(self.config.transformation_config.root_dir) / "train",
                valid_data_dir  = Path(self.config.transformation_config.root_dir) / "valid",  
            ),
            
            model_config = self._get_model_config(model_name=params.model_name),
        )