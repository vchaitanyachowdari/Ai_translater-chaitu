import os
import sys
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import ProjectConfig
    from src.data_processing.dataset import TranslationDataset
    from src.model.trainer import TranslationTrainer
    from src.model.evaluator import TranslationEvaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {PROJECT_ROOT}")
    sys.exit(1)

import logging
from src.data_processing.dataset_builder import DatasetBuilder
from src.optimization.hyperparameter_tuner import HyperparameterTuner
from src.optimization.data_augmentation import DataAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = ProjectConfig()
    
    # Create datasets
    logger.info("Creating datasets...")
    dataset_builder = DatasetBuilder(config)
    train_dataset, val_dataset, test_dataset = dataset_builder.create_datasets()
    
    # Perform hyperparameter tuning
    tuner = HyperparameterTuner(
        model_class=TranslationModel,
        dataset=train_dataset,
        config=config,
        n_trials=50  # Adjust based on your computational resources
    )
    best_params = tuner.run_optimization()
    
    # Update config with best parameters
    for key, value in best_params.items():
        setattr(config, key.upper(), value)
    
    # Data augmentation
    augmenter = DataAugmenter(model, tokenizer, config.DEVICE)
    
    # Perform back-translation augmentation
    augmented_texts = augmenter.back_translate(
        train_dataset.source_texts[:1000]  # Augment a subset of training data
    )
    
    # Add augmented data to training set
    train_dataset.extend(augmented_texts)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = TranslationTrainer(config)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_dataloader, val_dataloader)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = TranslationEvaluator(
        trainer.model,
        trainer.tokenizer,
        trainer.device
    )
    results = evaluator.evaluate(test_dataloader)
    
    logger.info(f"Final BLEU score: {results['bleu']:.2f}")

if __name__ == "__main__":
    main()