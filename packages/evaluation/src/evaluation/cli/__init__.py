import typer
from dotenv import load_dotenv
from core.logging import logger

load_dotenv()
app = typer.Typer()

@app.command("eval")
def eval(
    model: str = typer.Option("xgboost", help="Model: [xgboost, lstm, itransformer]"),
    sliding_window_size: int = typer.Option(6, help=""),
    sliding_horizon: int = typer.Option(1, help=""),
):
    from evaluation.config import ConfigurationManager
    from evaluation.components import Evaluation
    from core.mlflow_setup import setup_mlflow
    from dataclasses import replace
    import mlflow
    
    setup_mlflow()

    config_manager = ConfigurationManager()
    config         = config_manager.get_evaluation_config()

    config = replace(
        config,
        model = replace(
            config.model, 
            model_name  = model,
            window_size = sliding_window_size,
            horizon     = sliding_horizon,
        ),
    )
    
    evaluation = Evaluation(config=config)
    with mlflow.start_run(run_name=model):
        evaluation.evaluate()

if __name__ == "__main__":
    app()