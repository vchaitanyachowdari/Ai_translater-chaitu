import os
import torch
from src.models.multilingual_translator import MultilingualTranslator
from src.data.translation_dataset import TranslationDataset
from config.model_config import ModelConfig
from transformers import AdamW

def load_data():
    # Load your new training data here
    # For example, from a CSV file or a database
    source_texts = [...]  # New source texts
    target_texts = [...]  # New target texts
    return source_texts, target_texts

def retrain_model():
    # Load configuration
    config = ModelConfig()
    
    # Load new data
    source_texts, target_texts = load_data()
    
    # Initialize model and tokenizer
    model = MultilingualTranslator(config)
    tokenizer = model.tokenizer
    
    # Create dataset and dataloader
    dataset = TranslationDataset(source_texts, target_texts, tokenizer, config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                labels=batch['target_ids']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # Save the retrained model
    model.save("models/retrained_model")

if __name__ == "__main__":
    retrain_model() 