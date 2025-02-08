import torch

class TranslationAgent:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def translate(self, text):
        """Translate a single text input"""
        self.model.eval()
        with torch.no_grad():
            return self.model.translate(text)
    
    def batch_translate(self, texts):
        """Translate a batch of texts"""
        return [self.translate(text) for text in texts]
    
    def save_model(self, path):
        """Save the model state"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load a saved model state"""
        self.model.load_state_dict(torch.load(path)) 