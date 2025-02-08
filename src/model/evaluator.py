import torch
from sacrebleu import corpus_bleu
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationEvaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def translate_batch(self, batch):
        """Translate a batch of texts"""
        input_ids = batch['source_ids'].to(self.device)
        attention_mask = batch['source_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def evaluate(self, test_dataloader):
        """Evaluate the model on test data"""
        hypotheses = []
        references = []
        
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Generate translations
            translations = self.translate_batch(batch)
            hypotheses.extend(translations)
            
            # Get reference translations
            refs = self.tokenizer.batch_decode(
                batch['target_ids'],
                skip_special_tokens=True
            )
            references.extend(refs)
        
        # Calculate BLEU score
        bleu = corpus_bleu(hypotheses, [references])
        
        # Log results
        logger.info(f"BLEU score: {bleu.score:.2f}")
        
        # Sample translations
        num_samples = min(5, len(hypotheses))
        logger.info("\nSample translations:")
        for i in range(num_samples):
            logger.info(f"\nSource: {references[i]}")
            logger.info(f"Translation: {hypotheses[i]}")
        
        return {
            'bleu': bleu.score,
            'hypotheses': hypotheses,
            'references': references
        } 