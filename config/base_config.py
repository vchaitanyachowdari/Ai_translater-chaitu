from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class BaseConfig:
    # System Configuration
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4
    DEBUG: bool = False
    
    # Path Configuration
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    OUTPUT_DIR: str = "outputs"
    LOG_DIR: str = "logs"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    WANDB_PROJECT: str = "ai-translator"
    
    @classmethod
    def create(cls, **kwargs):
        """Create a new config instance with updated values"""
        return cls(**kwargs) 