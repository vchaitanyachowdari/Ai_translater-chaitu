import optuna
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any
import wandb
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, model_class, dataset, config, n_trials=100):
        self.model_class = model_class
        self.dataset = dataset
        self.config = config
        self.n_trials = n_trials
        self.device = torch.device(config.DEVICE)
        self.best_params = None
        self.best_score = float('-inf')
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'num_epochs': trial.suggest_int('num_epochs', 2, 10),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 2000),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 1.0)
        }
        
        # Create data loaders with current batch size
        train_loader, val_loader = self._create_dataloaders(params['batch_size'])
        
        # Initialize model with current parameters
        model = self.model_class(
            self.config.SOURCE_LANG,
            self.config.TARGET_LANG,
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Train model with current parameters
        score = self._train_and_evaluate(model, train_loader, val_loader, params)
        
        # Update best parameters if necessary
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            
        return score
    
    def _create_dataloaders(self, batch_size: int) -> tuple:
        """Create train and validation dataloaders"""
        # Split dataset into train and validation
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        return train_loader, val_loader
    
    def _train_and_evaluate(self, model, train_loader, val_loader, params: Dict[str, Any]) -> float:
        """Train model with given parameters and return validation score"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Initialize wandb for tracking
        wandb.init(
            project="nmt-hyperparameter-tuning",
            config=params,
            reinit=True
        )
        
        best_val_loss = float('inf')
        for epoch in range(params['num_epochs']):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['source_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            val_loss = self._evaluate(model, val_loader)
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': total_loss / len(train_loader),
                'val_loss': val_loss
            })
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        wandb.finish()
        return -best_val_loss  # Return negative because Optuna minimizes
    
    def _evaluate(self, model, val_loader) -> float:
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['source_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                
                outputs = model(input_ids, attention_mask, labels=labels)
                total_loss += outputs.loss.item()
        
        return total_loss / len(val_loader)
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        
        return trial.params 