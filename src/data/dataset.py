from torch.utils.data import Dataset
from typing import List, Dict, Any
import torch

class TranslationDataset(Dataset):
    """Dataset for translation tasks"""
    
    def __init__(self, 
                 source_texts: List[str], 
                 target_texts: List[str], 
                 tokenizer: Any, 
                 config: Any):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self) -> int:
        return len(self.source_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source = self.source_texts[idx]
        target = self.target_texts[idx]
        
        # Tokenize inputs
        source_encoding = self._tokenize(source)
        target_encoding = self._tokenize(target)
        
        return {
            'source_ids': source_encoding['input_ids'].squeeze(),
            'source_mask': source_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_mask': target_encoding['attention_mask'].squeeze()
        }
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text with proper padding and truncation"""
        return self.tokenizer(
            text,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) 