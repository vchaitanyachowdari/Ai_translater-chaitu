from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

class MultilingualTranslator:
    def __init__(self, config):
        self.config = config
        # Initialize M2M100 model which supports 100+ languages
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
    
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