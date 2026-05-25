import typer
from dotenv import load_dotenv
from core.logging import logger

load_dotenv()
app = typer.Typer()

@app.command("train")
def train(
    model: str = typer.Option("xgboost", help="Model: [xgboost, lstm, itransformer]"),
    sliding_window_size: int = typer.Option(6, help=""),
    sliding_horizon: int = typer.Option(1, help=""),
):
    from training.config import ConfigurationManager
    from training.components import Training
    from core.mlflow_setup import setup_mlflow
    from dataclasses import replace
    import mlflow
    
    setup_mlflow()

    config_manager = ConfigurationManager()
    config         = config_manager.get_training_config()

    config = replace(
        config,
        model = replace(
            config.model, 
            model_name  = model,
            window_size = sliding_window_size,
            horizon     = sliding_horizon,
        ),
    )
    
    training = Training(config=config)
    with mlflow.start_run(run_name=model):
        training.train()
        training._upload_model()
        training._upload_encoders()

if __name__ == "__main__":
    app()