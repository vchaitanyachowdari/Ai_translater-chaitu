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
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translated_text[0] 