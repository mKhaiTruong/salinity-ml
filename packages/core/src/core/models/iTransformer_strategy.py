import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from core.models import BaseModelStrategy
from core.models.configs import ModelConfig, iTransformerConfig

class iTransformerStrategy(BaseModelStrategy):
    def __init__(self, config: iTransformerConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model  = self._build_model().to(self.device)
        self.criterion  = nn.MSELoss()
        self.optimizer  = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def _build_model(self):
        class iTransformer(nn.Module):
            def __init__(self, window_size, hidden_size, horizon):
                super().__init__()
                self.norm1  = nn.LayerNorm(window_size)
                self.attn   = nn.MultiheadAttention(
                    embed_dim = window_size, 
                    num_heads = 2, 
                    batch_first = True
                )
                self.norm2  = nn.LayerNorm(window_size)
                self.ff     = nn.Sequential(
                    nn.Linear(window_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, window_size)
                )
                
                self.fc = nn.Linear(window_size, horizon)
                
            def forward(self, x):
                '''
                    x:   (batch, 6, 9)   < (batch, window_size, n_features)
                    x:   (batch, 9, 6)   < after transpose, embed_dim = window_size = 6
                    out: (batch, 9, 6)   < after attention
                    out: (batch, 9, 6)   < after ff
                    out: (batch, 6)      < after mean(dim=1), collapse dim 9
                    out: (batch, horizon) < after fc
                '''
                x   = torch.transpose(x, 1, 2)
                attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
                out = x + attn_out
                out = out + self.ff(self.norm2(out))
                out = torch.mean(out, dim=1)
                return self.fc(out)
                
        return iTransformer(
            hidden_size = self.config.hidden_size,
            window_size = self.model_config.window_size,
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
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()
    
    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path / "model.pt")
        
        import json
        metadata = {}
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
        
    def save_onnx(self, path: Path) -> None:
        dummy_input = torch.randn(
            1, 
            self.model_config.window_size,
            self._input_size
        ).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            path / "model.onnx",
            input_names  = ["input"],
            output_names = ["output"],
            dynamic_axes = {
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )