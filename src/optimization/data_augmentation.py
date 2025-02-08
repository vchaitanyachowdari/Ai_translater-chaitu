import torch
from typing import List, Tuple
import random
from tqdm import tqdm

class DataAugmenter:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def back_translate(self, texts: List[str], intermediate_lang: str = "fr") -> List[str]:
        """
        Perform back-translation augmentation:
        source -> intermediate -> source
        """
        self.model.eval()
        augmented_texts = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Back-translating"):
                # Translate to intermediate language
                intermediate = self._translate(text, intermediate_lang)
                
                # Translate back to source language
                back_translated = self._translate(intermediate, "en")
                
                augmented_texts.append(back_translated)
        
        return augmented_texts
    
    def _translate(self, text: str, target_lang: str) -> str:
        """Helper function to translate text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def synonym_replacement(self, texts: List[str], n_replacements: int = 1) -> List[str]:
        """
        Replace words with their synonyms using WordNet
        """
        from nltk.corpus import wordnet
        import nltk
        nltk.download('wordnet')
        
        augmented_texts = []
        
        for text in texts:
            words = text.split()
            n = min(n_replacements, len(words))
            positions = random.sample(range(len(words)), n)
            
            for pos in positions:
                word = words[pos]
                synonyms = []
                
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.append(lemma.name())
                
                if synonyms:
                    words[pos] = random.choice(synonyms)
            
            augmented_texts.append(' '.join(words))
        
        return augmented_texts
    
    def random_swap(self, texts: List[str], n_swaps: int = 1) -> List[str]:
        """
        Randomly swap words in the text
        """
        augmented_texts = []
        
        for text in texts:
            words = text.split()
            for _ in range(n_swaps):
                if len(words) >= 2:
                    pos1, pos2 = random.sample(range(len(words)), 2)
                    words[pos1], words[pos2] = words[pos2], words[pos1]
            
            augmented_texts.append(' '.join(words))
        
        return augmented_texts 