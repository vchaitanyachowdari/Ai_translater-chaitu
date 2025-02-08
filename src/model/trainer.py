import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {config.MODEL_NAME}")
        self.model = MarianMTModel.from_pretrained(config.MODEL_NAME)
        self.tokenizer = MarianTokenizer.from_pretrained(config.MODEL_NAME)
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
    def train(self, train_dataloader, val_dataloader):
        """Train the model"""
        # Initialize wandb for experiment tracking
        wandb.init(project="nmt-training", config=self.config.__dict__)
        
        num_training_steps = len(train_dataloader) * self.config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=num_training_steps
        )
        
        best_val_loss = float('inf')
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            self.model.train()
            total_train_loss = 0
            train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in train_iterator:
                self.optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['source_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                
                train_iterator.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            # Validation
            val_loss = self.evaluate(val_dataloader)
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model')
                
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['source_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)
    
    def save_model(self, name):
        """Save the model"""
        save_dir = Path(self.config.MODEL_DIR) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}") 