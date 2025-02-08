import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            input_ids = batch['source_ids'].to(self.device)
            attention_mask = batch['source_mask'].to(self.device)
            labels = batch['target_ids'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader) 