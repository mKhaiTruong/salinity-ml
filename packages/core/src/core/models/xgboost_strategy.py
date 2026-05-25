import joblib
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from core.models import BaseModelStrategy
from core.models.configs import XGBoostConfig

class XGBoostStrategy(BaseModelStrategy):
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.model  = self._build_model()
    
    def _build_model(self):
        base_model = XGBRegressor(
            n_estimators    = self.config.n_estimators,
            learning_rate   = self.config.learning_rate,
            max_depth       = self.config.max_depth,
            subsample       = self.config.subsample,
            colsample_bytree= 0.8,
            objective       = "reg:squarederror",
            random_state    = 42,
            n_jobs          = -1
        )
        return MultiOutputRegressor(base_model)
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None) -> None:
        self.model.fit(X_train, y_train)
    
    def validate(self, X_valid, y_valid) -> dict:
        preds = self.model.predict(X_valid)
        
        mae  = mean_absolute_error(y_valid, preds)
        rmse = root_mean_squared_error(y_valid, preds)
        r2   = r2_score(y_valid, preds)

        return {
            "valid_mae": float(mae),
            "valid_rmse": float(rmse),
            "valid_r2": float(r2),
        }
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path: Path) -> None:
        joblib.dump(self.model, path / "model.pkl")
        
        import json
        metadata = {}
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    def load(self, path: Path) -> None:
        self.model = joblib.load(path / "model.pkl")
        
    def save_onnx(self, path: Path) -> None:
        pass  # XGBoost with MultiOutputRegressor does not support single-file ONNX export