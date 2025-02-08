import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

class TranslationModel(nn.Module):
    def __init__(self, source_lang, target_lang):
        super().__init__()
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits
    
    def translate(self, text):
        """Translate a single text input"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translated_text[0]
    
    def batch_translate(self, texts):
        """Translate a batch of texts"""
        # Tokenize all texts at once
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Generate translations
        translated = self.model.generate(**inputs)
        
        # Decode all translations
        translated_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return translated_texts 