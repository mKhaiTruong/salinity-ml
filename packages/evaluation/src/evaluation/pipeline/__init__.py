import sys
import mlflow

from core.exception import CustomException
from core.mlflow_setup import setup_mlflow

from evaluation.config import ConfigurationManager
from evaluation.components import Evaluation

setup_mlflow()

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        cfg_manager = ConfigurationManager()
        config      = cfg_manager.get_evaluation_config()
        evaluation  = Evaluation(config=config)
        
        with mlflow.start_run(run_name=config.model.model_name):
            evaluation.evaluate()