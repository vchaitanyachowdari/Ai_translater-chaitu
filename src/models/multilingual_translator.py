from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from huggingface_hub import login
import torch
from .base_model import BaseModel

class MultilingualTranslator(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Authenticate if necessary
        if config.HUGGINGFACE_TOKEN:
            login(token=config.HUGGINGFACE_TOKEN)
        
        # Load model with authentication
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            config.MODEL_NAME,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            config.MODEL_NAME,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None
        )
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for training"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between any supported language pair"""
        # Set source language
        self.tokenizer.src_lang = source_lang
        
        # Encode input text
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang)
        )
        
        # Decode translation
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
    
    def translate_to_spanish(self, text: str) -> str:
        """Translate English to Spanish"""
        return self.translate(text, source_lang="en", target_lang="es")
    
    def translate_to_hindi(self, text: str) -> str:
        """Translate English to Hindi"""
        return self.translate(text, source_lang="en", target_lang="hi")
    
    def translate_from_spanish(self, text: str) -> str:
        """Translate Spanish to English"""
        return self.translate(text, source_lang="es", target_lang="en")
    
    def translate_from_hindi(self, text: str) -> str:
        """Translate Hindi to English"""
        return self.translate(text, source_lang="hi", target_lang="en")
    
    def save(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load model and tokenizer"""
        self.model = M2M100ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = M2M100Tokenizer.from_pretrained(path)
        self.model.to(self.device) 