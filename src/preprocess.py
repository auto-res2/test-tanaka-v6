import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticExpertDataset(Dataset):
    def __init__(self, num_samples=500, state_dim=10, action_dim=2):
        super(SyntheticExpertDataset, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = torch.rand(num_samples, state_dim)
        weight = torch.randn(state_dim, action_dim)
        bias = torch.randn(action_dim)
        self.actions = self.states @ weight + bias + 0.05 * torch.randn(num_samples, action_dim)
    
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index]

def add_gaussian_noise(state, noise_std=0.1):
    noise = torch.randn_like(state) * noise_std
    return state + noise

def create_dataloader(num_samples=500, state_dim=10, action_dim=2, batch_size=32):
    dataset = SyntheticExpertDataset(num_samples, state_dim, action_dim)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
