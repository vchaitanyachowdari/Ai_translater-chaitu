import os
import pandas as pd
import torch
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from langdetect import detect
from transformers import MarianTokenizer
from torch.utils.data import Dataset, random_split
from config import ProjectConfig

class DatasetBuilder:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.tokenizer = MarianTokenizer.from_pretrained(config.MODEL_NAME)
        
    def load_opus_data(self, opus_path: str) -> pd.DataFrame:
        """Load parallel data from OPUS dataset"""
        data = []
        with open(os.path.join(opus_path, f"{self.config.SOURCE_LANG}.txt"), 'r', encoding='utf-8') as source_file, \
             open(os.path.join(opus_path, f"{self.config.TARGET_LANG}.txt"), 'r', encoding='utf-8') as target_file:
            for source_line, target_line in zip(source_file, target_file):
                data.append({
                    'source': source_line.strip(),
                    'target': target_line.strip()
                })
        return pd.DataFrame(data)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Add more cleaning steps as needed
        return text
    
    def validate_language_pair(self, text: str, expected_lang: str) -> bool:
        """Verify text is in the expected language"""
        try:
            detected_lang = detect(text)
            return detected_lang == expected_lang
        except:
            return False
    
    def filter_sentence_pair(self, source: str, target: str) -> bool:
        """Apply filtering criteria to sentence pairs"""
        # Length checks
        if len(source.split()) > self.config.MAX_LENGTH or len(target.split()) > self.config.MAX_LENGTH:
            return False
        if len(source.split()) < 3 or len(target.split()) < 3:  # Skip very short sentences
            return False
            
        # Language validation
        if not self.validate_language_pair(source, self.config.SOURCE_LANG):
            return False
        if not self.validate_language_pair(target, self.config.TARGET_LANG):
            return False
            
        return True
    
    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the dataset"""
        processed_data = []
        
        for _, row in tqdm(raw_data.iterrows(), desc="Preprocessing data"):
            source = self.clean_text(row['source'])
            target = self.clean_text(row['target'])
            
            if self.filter_sentence_pair(source, target):
                processed_data.append({
                    'source': source,
                    'target': target
                })
                
        return pd.DataFrame(processed_data)
    
    def create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train, validation, and test datasets"""
        # Load and preprocess data
        raw_data = self.load_opus_data(os.path.join(self.config.DATA_DIR, 'opus'))
        processed_data = self.preprocess_data(raw_data)
        
        # Create dataset
        full_dataset = TranslationDataset(
            source_texts=processed_data['source'].tolist(),
            target_texts=processed_data['target'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_LENGTH
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * (1 - self.config.TEST_SIZE - self.config.VALID_SIZE))
        valid_size = int(total_size * self.config.VALID_SIZE)
        test_size = total_size - train_size - valid_size
        
        train_dataset, valid_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, valid_dataset, test_dataset 


config = ProjectConfig()
dataset_builder = DatasetBuilder(config)
train_dataset, valid_dataset, test_dataset = dataset_builder.create_datasets()