import os, sys, json, time, glob
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score
)

from core.logging import logger
from core.exception import CustomException
from core.data.process import preprocess
from core.data.sliding_window import sliding


from core.models.xgboost_strategy import XGBoostStrategy
from core.models.lstm_strategy import LSTMStrategy
from core.models.iTransformer_strategy import iTransformerStrategy

from evaluation import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    # ─────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────
    def _get_infer_data(self):
        train_files = glob.glob(os.path.join(Path(self.config.data_dir).parent / "train", "*.csv"))
        valid_files = glob.glob(os.path.join(Path(self.config.data_dir).parent / "valid", "*.csv"))
        infer_files = glob.glob(os.path.join(self.config.data_dir, "*.csv"))

        full_df = pd.concat([
            pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True),
            pd.concat([pd.read_csv(f) for f in valid_files], ignore_index=True),
            pd.concat([pd.read_csv(f) for f in infer_files], ignore_index=True),
        ], ignore_index=True)
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)

        clean_df = preprocess(
            full_df, out_dir=str(self.config.encoders_dir), fit_encoders=False
        )

        X, y, _, _ = sliding(
            clean_df,
            train_cutoff = pd.to_datetime(full_df["timestamp"]).max(),
            model_type   = self.config.model.model_name,
            window_size  = self.config.model.window_size,
            horizon      = self.config.model.horizon,
        )
        return X, y
    
    # ─────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────
    def _load_metadata(self) -> dict:
        path = self.config.model.model_dir / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"metadata.json not found at {path}")
        with open(path) as f:
            return json.load(f)
        
    def _load_strategy(self):
        name = self.config.model.model_name
        metadata = self._load_metadata()
        
        model_config = ModelConfig(
            model_name  = name,
            window_size = self.config.model.window_size,
            horizon     = self.config.model.horizon,
        )
        
        if name == "xgboost":
            strategy = XGBoostStrategy(self.config.model_config)
        elif name == "lstm":
            strategy = LSTMStrategy(self.config.model_config, model_config, input_size=metadata["input_size"])
        elif name == "iTransformer":
            strategy = iTransformerStrategy(self.config.model_config, model_config)
        else:
            raise ValueError(f"Unsupported model: {name}")
        
        strategy.load(self.config.model.model_dir)
        return strategy
        
    def _load_onnx(self, use_cuda: bool = False):
        import onnxruntime as ort
        path = self.config.model.onnx_dir / "model.onnx"
        if not path.exists():
            logger.warning(f"ONNX not found: {path}")
            return None
        
        if use_cuda:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                logger.warning("CUDA not available — skipping CUDA ONNX")
                return None
            return ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])

        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    # ─────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────
    def _compute_metrics(self, preds: np.ndarray, y: np.ndarray, elapsed_ms: float) -> dict:
        return {
            "mae":  round(float(mean_absolute_error(y, preds)), 4),
            "rmse": round(float(root_mean_squared_error(y, preds)), 4),
            "r2":   round(float(r2_score(y, preds)), 4),
            "ms":   round(elapsed_ms, 2),
        }
        
    # ─────────────────────────────────────────────
    # Eval Helpers
    # ─────────────────────────────────────────────   
    def _eval_native(self, X, y) -> dict:
        strategy = self._load_strategy()
        
        start    = time.perf_counter()
        preds    = strategy.predict(X)
        elapsed  = (time.perf_counter() - start) * 1000
        
        return self._compute_metrics(preds, y, elapsed)
    
    def _eval_onnx(self, X, y, use_cuda: bool = False) -> dict | None:
        session  = self._load_onnx(use_cuda)
        if session is None: 
            return None
        
        start      = time.perf_counter()
        ort_inputs = {session.get_inputs()[0].name: X.astype(np.float32)}
        preds      = session.run(None, ort_inputs)[0]
        elapsed    = (time.perf_counter() - start) * 1000
        
        return self._compute_metrics(preds, y, elapsed)
    
    def _print_table(self, results: dict):
        print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'ms':>10}")
        print("-" * 56)
        for name, m in results.items():
            print(f"{name:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m.get('ms', 0):>10.2f}")

    def _save_metrics(self, results: dict):
        path = self.config.root_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved -> {path}")
    
    
    # ─────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────
    def evaluate(self) -> dict:
        try:
            X, y    = self._get_infer_data()
            results = {}
            name    = self.config.model.model_name
            
            logger.info("Evaluating native model...")
            results["native"] = self._eval_native(X, y)
            
            if name != "xgboost":
                logger.info("Evaluating ONNX CPU...")
                onnx_cpu = self._eval_onnx(X, y, use_cuda=False)
                if onnx_cpu:
                    results["onnx_cpu"] = onnx_cpu

                logger.info("Evaluating ONNX CUDA...")
                onnx_cuda = self._eval_onnx(X, y, use_cuda=True)
                if onnx_cuda:
                    results["onnx_cuda"] = onnx_cuda
            
            self._print_table(results)
            self._save_metrics(results)
            return results
            
        except Exception as e:
            raise CustomException(e, sys)