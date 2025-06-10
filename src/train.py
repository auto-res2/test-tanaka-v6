"""
Training module for HFR-DT and FiT models.
Contains model definitions and training logic for the experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class HFR_DT_Model(nn.Module):
    """
    Hierarchical Flexible Resolution Diffusion Transformer (HFR-DT) model.
    A simplified implementation for experimental comparison.
    """
    def __init__(self, input_channels=3, hidden_dim=64):
        super(HFR_DT_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.adaptive_norm = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        loss = torch.mean(x)
        return loss

class FiT_Model(nn.Module):
    """
    Flexible Vision Transformer (FiT) baseline model.
    A simplified implementation for experimental comparison.
    """
    def __init__(self, input_channels=3, hidden_dim=64):
        super(FiT_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        loss = torch.mean(x)
        return loss

class HFR_DT_Model_WithHierarchy(nn.Module):
    """
    HFR-DT model with hierarchical token representation for multi-scale experiments.
    """
    def __init__(self, input_channels=3):
        super(HFR_DT_Model_WithHierarchy, self).__init__()
        self.coarse_conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.fine_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        coarse_tokens = self.relu(self.coarse_conv(x))
        fine_tokens = self.relu(self.fine_conv(coarse_tokens))
        final_output = torch.sigmoid(self.final_conv(fine_tokens))
        return coarse_tokens, fine_tokens, final_output

def train_model(model, dataloader, num_epochs=2, learning_rate=1e-4, device='cuda'):
    """
    Train a model for the specified number of epochs.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        tuple: Lists of losses and gradient norms during training
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    losses = []
    grad_norms = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item()
            
            optimizer.step()
            
            losses.append(loss.item())
            grad_norms.append(total_grad_norm)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Grad Norm={total_grad_norm:.4f}")
                
            if batch_idx >= 20:
                break
                
    return losses, grad_norms

def compare_models(hfr_model, fit_model, dataloader, num_epochs=2, device='cuda'):
    """
    Compare training dynamics between HFR-DT and FiT models.
    
    Args:
        hfr_model: HFR-DT model instance
        fit_model: FiT model instance
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        dict: Training results for both models
    """
    print("Training HFR-DT model...")
    hfr_losses, hfr_grad_norms = train_model(hfr_model, dataloader, num_epochs, device=device)
    
    print("Training FiT model...")
    fit_losses, fit_grad_norms = train_model(fit_model, dataloader, num_epochs, device=device)
    
    results = {
        'hfr_dt': {
            'losses': hfr_losses,
            'grad_norms': hfr_grad_norms
        },
        'fit': {
            'losses': fit_losses,
            'grad_norms': fit_grad_norms
        }
    }
    
    return results

if __name__ == "__main__":
    from preprocess import get_cifar10_dataloader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader = get_cifar10_dataloader(batch_size=32, train=True, download=True)
    
    hfr_model = HFR_DT_Model()
    fit_model = FiT_Model()
    
    results = compare_models(hfr_model, fit_model, dataloader, num_epochs=1, device=device)
    
    print("Training completed!")
    print(f"HFR-DT final loss: {results['hfr_dt']['losses'][-1]:.4f}")
    print(f"FiT final loss: {results['fit']['losses'][-1]:.4f}")
