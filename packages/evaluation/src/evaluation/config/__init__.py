from core.constants import *
from core import read_yaml, create_directories

from core.models.configs import (
    ModelConfig, XGBoostConfig, LSTMConfig, iTransformerConfig
)
from evaluation import EvaluationModelConfig, EvaluationConfig, ModelSpecificConfig

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
        
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config  = self.config.evaluation_config
        train_config = self.config.training_config
        train_params = self.params.training_params
        create_directories([eval_config.root_dir])
        
        return EvaluationConfig(
            root_dir = Path(eval_config.root_dir),
            encoders_dir = Path(train_config.encoders_dir),
            data_dir = Path(self.config.transformation_config.root_dir) / "infer",
            
            model    = EvaluationModelConfig(
                model_name  = train_params.model_name,
                window_size = train_params.sliding_window_size,
                horizon     = train_params.sliding_horizon,
                model_dir   = Path(train_config.root_dir) / train_params.model_name,
                onnx_dir    = Path(train_config.root_dir) / train_params.model_name,
            ),
            
            model_config = self._get_model_config(model_name=train_params.model_name),
        )