from dataclasses import dataclass
from .base_config import BaseConfig
from typing import Optional
import os

@dataclass
class ModelConfig(BaseConfig):
    # Authentication
    HUGGINGFACE_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Model Configuration
    MODEL_NAME: str = "facebook/m2m100_418M"
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 32
    
    # Training Configuration
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 5
    WARMUP_STEPS: int = 1000
    GRADIENT_CLIP: float = 1.0
    WEIGHT_DECAY: float = 0.01
    
    # Audio Configuration
    SAMPLE_RATE: int = 16000
    ASR_MODEL: str = "Harveenchadha/wav2vec2-large-xlsr-hindi"
    TTS_MODEL: str = "facebook/mms-tts-hin" 