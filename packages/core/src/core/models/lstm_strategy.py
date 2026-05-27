import joblib
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score
)

from core.models import BaseModelStrategy
from core.models.configs import ModelConfig, LSTMConfig

class LSTMStrategy(BaseModelStrategy):
    def __init__(self, config: LSTMConfig, model_config: ModelConfig, input_size: int):
        self.config = config
        self.model_config = model_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model      = self._build_model(input_size).to(self.device)
        self.criterion  = nn.MSELoss()
        self.optimizer  = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def _build_model(self, input_size: int):
        class SalinityLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size  = input_size,
                    hidden_size = hidden_size,
                    num_layers  = num_layers,
                    dropout     = dropout,
                    batch_first = True
                )
                self.fc = nn.Linear(hidden_size, horizon)
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out    = out[:, -1, :]
                return self.fc(out)
            
        return SalinityLSTM(
            input_size  = input_size,
            hidden_size = self.config.hidden_size,
            num_layers  = self.config.num_layers,
            dropout     = self.config.dropout,
            horizon     = self.model_config.horizon
        )
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None) -> None:
        self._input_size = X_train.shape[-1]    # save this for ONNX
        
        batch_size  = self.config.batch_size
        patience    = self.config.patience
        best_loss   = float('inf')
        no_improve  = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for i in range(0, len(X_train), batch_size):
                inputs  = torch.from_numpy(X_train[i:i+batch_size]).float().to(self.device)
                targets = torch.from_numpy(y_train[i:i+batch_size]).float().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X_train) / batch_size)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            
            if X_valid is not None:
                val_preds   = self.predict(X_valid)
                val_loss    = np.mean((val_preds - y_valid) ** 2)
                monitor_loss = val_loss
            else:
                monitor_loss = avg_loss

            if monitor_loss < best_loss:
                best_loss  = monitor_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def validate(self, X_valid, y_valid) -> dict:
        preds = self.predict(X_valid)
        return {
            "valid_mae":  float(mean_absolute_error(y_valid, preds)),
            "valid_rmse": float(root_mean_squared_error(y_valid, preds)),
            "valid_r2":   float(r2_score(y_valid, preds)),
        }
    
    def predict(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()
    
    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path / "model.pt")
        
        import json
        metadata = {"input_size": self._input_size}
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
    
    def save_onnx(self, path: Path) -> None:
        import torch
        
        dummy_input = torch.randn(
            1, 
            self.model_config.window_size,
            self._input_size
        ).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            path / "model.onnx",
            dynamo       = False,
            input_names  = ["input"],
            output_names = ["output"],
            dynamic_axes = {
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )