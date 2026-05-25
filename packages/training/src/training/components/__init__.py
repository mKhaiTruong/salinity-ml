import os, glob, json, time, boto3, mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import asdict

from core.logging import logger
from core.data.process import preprocess
from core.data.sliding_window import sliding
from core.models import BaseModelStrategy

from training import TrainingConfig
from training.strategy_manager import get_strategy

class Training:
    def __init__(self, config: TrainingConfig):
        self.config       = config
        self.model_config = config.model_config

        self.clean_df, self.train_cutoff = self._get_clean_df()
        self.X_train, self.y_train, self.X_valid, self.y_valid = self._get_windows()
        self.model = self._get_model()
        self.s3    = boto3.client("s3")
        
    # ─────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────
    def _get_clean_df(self):
        train_files = glob.glob(os.path.join(self.config.data.train_data_dir, "*.csv"))
        valid_files = glob.glob(os.path.join(self.config.data.valid_data_dir, "*.csv"))
        
        train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
        valid_df = pd.concat([pd.read_csv(f) for f in valid_files], ignore_index=True)
        
        full_df = pd.concat([train_df, valid_df], ignore_index=True)
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)
        
        train_cutoff = train_df["timestamp"].max()
        logger.info(f"Train cutoff: {train_cutoff}")
        
        clean_df = preprocess(full_df, out_dir=str(self.config.encoders_dir), fit_encoders=True)
        return clean_df, train_cutoff
    
    def _get_windows(self):
        logger.info("Preparing sliding windows...")
        return sliding(
            self.clean_df, self.train_cutoff,
            model_type  = self.config.model.model_name,
            window_size = self.config.model.window_size,
            horizon     = self.config.model.horizon,
        )
        
    
    # ─────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────
    def _get_model(self) -> BaseModelStrategy:
        name = self.config.model.model_name
        if name == "lstm":
            return get_strategy(self.config, input_size=self.X_train.shape[-1])
        return get_strategy(self.config)
    
      
    # ─────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────
    def train(self):
        mlflow.log_params(self._get_log_params())
        
        # ── training ─────────────────────────────────────────
        logger.info("Training model...")

        start   = time.time()
        self.model.train(self.X_train, self.y_train)
        elapsed = time.time() - start
        
        logger.info(f"Training done in {elapsed:.2f}s")
        
        # ── Validation ─────────────────────────────────
        metrics = self.model.validate(self.X_valid, self.y_valid)
        logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")
        mlflow.log_metrics(metrics)
        
        # ── Saving weights ─────────────────────────────────
        self.model.save(self.config.model_dir)

        # ── Optim - ONNX ─────────────────────────────────
        self.model.save_onnx(self.config.model_dir)

        mlflow.log_artifacts(str(self.config.model_dir))
        return metrics
    
    
    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────
    def _get_log_params(self) -> dict:
        return {
            "model_name":  self.config.model.model_name,
            "window_size": self.config.model.window_size,
            "horizon":     self.config.model.horizon,
            **{k: v for k, v in asdict(self.model_config).items()},
        }
    
    def _upload_model(self):
        bucket     = os.environ["S3_BUCKET"]
        model_name = self.config.model.model_name
        local_dir  = self.config.model_dir
        
        files_to_upload = list(local_dir.glob("*"))
        
        for file_path in files_to_upload:
            s3_key = f"artifacts/{model_name}/{file_path.name}"
            self.s3.upload_file(str(file_path), bucket, s3_key)
            logger.info(f"Uploaded {file_path.name} → s3://{bucket}/{s3_key}")
    
    def _upload_encoders(self):
        bucket   = os.environ["S3_BUCKET"]
        enc_dir  = self.config.encoders_dir

        for file_path in enc_dir.glob("*.pkl"):
            s3_key = f"artifacts/encoders/{file_path.name}"
            self.s3.upload_file(str(file_path), bucket, s3_key)
            logger.info(f"Uploaded {file_path.name} → s3://{bucket}/{s3_key}")