import sys
import mlflow

from core.exception import CustomException
from core.mlflow_setup import setup_mlflow

from training.config import ConfigurationManager
from training.components import Training

setup_mlflow()

class TrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        cfg_manager = ConfigurationManager()
        config      = cfg_manager.get_training_config()
        training    = Training(config=config)
        
        with mlflow.start_run(run_name=config.model.model_name):
            training.train()
            training._upload_model()
            training._upload_encoders()