from abc import ABC, abstractmethod
from pathlib import Path

class BaseModelStrategy(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_valid=None, y_valid=None) -> None: pass
    @abstractmethod
    def validate(self, X_valid, y_valid) -> dict: pass
    @abstractmethod
    def predict(self, X): pass
    @abstractmethod
    def save(self, path: Path) -> None: pass
    @abstractmethod
    def load(self, path: Path) -> None: pass
    @abstractmethod
    def save_onnx(self, path: Path) -> None: pass