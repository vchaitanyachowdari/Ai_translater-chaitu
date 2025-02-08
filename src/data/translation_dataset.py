import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from ..utils.exceptions import DataError
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class TranslationDataset(Dataset):
    """Dataset for machine translation with comprehensive error handling and validation"""
    
    def __init__(
        self, 
        source_texts: List[str], 
        target_texts: List[str], 
        tokenizer: Any, 
        config: Any,
        is_train: bool = True,
        max_length: Optional[int] = None
    ):
        """
        Initialize the translation dataset with validation.
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            tokenizer: Tokenizer instance
            config: Configuration object
            is_train: Whether this dataset is for training
            max_length: Optional override for max sequence length
        
        Raises:
            DataError: If there's an issue with the input data
        """
        self._validate_inputs(source_texts, target_texts)
        
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.max_length = max_length or config.MAX_LENGTH
        
        # Validate a sample batch to catch any issues early
        try:
            self._validate_batch(self[0])
        except Exception as e:
            raise DataError(f"Failed to process sample batch: {str(e)}")
        
        logger.info(f"Created dataset with {len(self)} examples")
    
    def _validate_inputs(self, source_texts: List[str], target_texts: List[str]) -> None:
        """Validate input data"""
        if not source_texts or not target_texts:
            raise DataError("Empty source or target texts")
            
        if len(source_texts) != len(target_texts):
            raise DataError(
                f"Mismatched source and target lengths: {len(source_texts)} vs {len(target_texts)}"
            )
        
        # Validate text content
        for i, (src, tgt) in enumerate(zip(source_texts, target_texts)):
            if not isinstance(src, str) or not isinstance(tgt, str):
                raise DataError(f"Non-string input at index {i}")
            if not src.strip() or not tgt.strip():
                raise DataError(f"Empty string at index {i}")
    
    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate batch structure and content"""
        required_keys = {'source_ids', 'source_mask', 'target_ids', 'target_mask'}
        if not all(k in batch for k in required_keys):
            raise DataError(f"Missing required keys in batch. Expected {required_keys}")
        
        # Validate tensor shapes
        for key in required_keys:
            if not isinstance(batch[key], torch.Tensor):
                raise DataError(f"Expected tensor for {key}, got {type(batch[key])}")
            if batch[key].dim() != 1:
                raise DataError(f"Expected 1D tensor for {key}, got shape {batch[key].shape}")
    
    def __len__(self) -> int:
        return len(self.source_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single translation pair with error handling"""
        try:
            source = self.source_texts[idx]
            target = self.target_texts[idx]
            
            # Tokenize with error handling
            source_encoding = self._safe_tokenize(source)
            target_encoding = self._safe_tokenize(target)
            
            batch = {
                'source_ids': source_encoding['input_ids'].squeeze(),
                'source_mask': source_encoding['attention_mask'].squeeze(),
                'target_ids': target_encoding['input_ids'].squeeze(),
                'target_mask': target_encoding['attention_mask'].squeeze(),
                'source_text': source,
                'target_text': target
            }
            
            self._validate_batch(batch)
            return batch
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            if self.is_train:
                # During training, return a valid but empty batch
                return self._get_empty_encoding()
            else:
                # During inference, we want to know about errors
                raise DataError(f"Failed to process item {idx}: {str(e)}")
    
    def _safe_tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text with error handling"""
        try:
            return self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length' if self.is_train else 'longest',
                truncation=True,
                return_tensors='pt'
            )
        except Exception as e:
            raise DataError(f"Tokenization failed: {str(e)}")
    
    def _get_empty_encoding(self) -> Dict[str, torch.Tensor]:
        """Return empty tensors of correct shape"""
        return {
            'source_ids': torch.zeros(self.max_length, dtype=torch.long),
            'source_mask': torch.zeros(self.max_length, dtype=torch.long),
            'target_ids': torch.zeros(self.max_length, dtype=torch.long),
            'target_mask': torch.zeros(self.max_length, dtype=torch.long),
            'source_text': '',
            'target_text': ''
        }
    
    @classmethod
    def create_dataloaders(
        cls,
        source_texts: List[str],
        target_texts: List[str],
        tokenizer: Any,
        config: Any,
        train_split: float = 0.8,
        valid_split: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders with error handling"""
        try:
            # Validate split ratios
            if not 0 < train_split < 1 or not 0 < valid_split < 1:
                raise DataError("Invalid split ratios")
            if train_split + valid_split >= 1:
                raise DataError("Split ratios sum to ≥1")
            
            total = len(source_texts)
            train_size = int(total * train_split)
            valid_size = int(total * valid_split)
            test_size = total - train_size - valid_size
            
            if any(size <= 0 for size in [train_size, valid_size, test_size]):
                raise DataError("Split resulted in empty dataset")
            
            # Create datasets
            datasets = []
            for i, (size, is_train) in enumerate([
                (train_size, True),
                (valid_size, False),
                (test_size, False)
            ]):
                start_idx = sum(len(d.source_texts) for d in datasets)
                datasets.append(cls(
                    source_texts[start_idx:start_idx + size],
                    target_texts[start_idx:start_idx + size],
                    tokenizer,
                    config,
                    is_train=is_train
                ))
            
            # Create dataloaders with error handling
            loaders = []
            for dataset, is_train in zip(datasets, [True, False, False]):
                batch_size = config.BATCH_SIZE if is_train else config.EVAL_BATCH_SIZE
                loaders.append(DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=is_train,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=True,
                    drop_last=is_train,
                    collate_fn=cls._safe_collate_fn
                ))
            
            return tuple(loaders)
            
        except Exception as e:
            raise DataError(f"Failed to create dataloaders: {str(e)}")
    
    @staticmethod
    def _safe_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function with error handling"""
        try:
            return {
                key: torch.stack([item[key] for item in batch])
                for key in batch[0].keys()
                if isinstance(batch[0][key], torch.Tensor)
            }
        except Exception as e:
            raise DataError(f"Collation failed: {str(e)}")