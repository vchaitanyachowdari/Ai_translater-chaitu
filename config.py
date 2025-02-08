import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjectConfig:
    # Language Configuration
    SOURCE_LANG: str = "hi"  # Hindi
    TARGET_LANG: str = "en"  # English
    
    # Domain Configuration
    DOMAIN: str = "general"  # Can be 'general', 'technical', 'news', 'conversational'
    
    # Performance Goals
    MIN_BLEU_SCORE: float = 30.0  # Minimum acceptable BLEU score
    MAX_LATENCY_MS: int = 1000    # Maximum acceptable latency in milliseconds
    
    # Resource Configuration
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_GPU_MEMORY_GB: Optional[float] = 8.0 if torch.cuda.is_available() else None
    NUM_WORKERS: int = 4  # For data loading
    
    # Model Configuration
    MODEL_NAME: str = "Helsinki-NLP/opus-mt-hi-en"  # Hugging Face model ID
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 32
    
    # Training Configuration
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 5
    WARMUP_STEPS: int = 1000
    GRADIENT_ACCUMULATION_STEPS: int = 1
    
    # Paths
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    CACHE_DIR: str = ".cache"
    LOGS_DIR: str = "logs"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Evaluation Configuration
    EVAL_BATCH_SIZE: int = 64
    TEST_SIZE: float = 0.1
    VALID_SIZE: float = 0.1 