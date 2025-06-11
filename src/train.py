import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

try:
    from pot import wasserstein
except ImportError:
    def wasserstein(X, Y, num_projections=50):
        return np.linalg.norm(np.mean(X, axis=0) - np.mean(Y, axis=0))

def ot_loss(expert_pairs, generated_pairs, num_projections=50):
    return wasserstein(expert_pairs.detach().cpu().numpy(), generated_pairs.detach().cpu().numpy(), num_projections=num_projections)

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_loss(self, state_action):
        score = self.net(state_action)
        loss = -score.mean()
        return loss

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.actor(state)

class OT_DBC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, ot_lambda=1.0):
        super(OT_DBC, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.diffusion = DiffusionModel(state_dim + action_dim, hidden_dim)
        self.ot_lambda = ot_lambda

    def forward(self, state):
        return self.policy(state)

    def compute_loss(self, state, expert_action):
        pred_action = self.policy(state)
        bc_loss = F.mse_loss(pred_action, expert_action)
        
        expert_sa = torch.cat([state, expert_action], dim=1)
        diffusion_loss = self.diffusion.compute_loss(expert_sa)
        
        generated_action = pred_action.detach()
        generated_sa = torch.cat([state, generated_action], dim=1)
        ot_loss_val = ot_loss(expert_sa, generated_sa)
        
        total_loss = bc_loss + diffusion_loss + self.ot_lambda * ot_loss_val
        return total_loss, {'bc_loss': bc_loss.item(),
                            'diffusion_loss': diffusion_loss.item(),
                            'ot_loss': ot_loss_val}

class Diffusion_BC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Diffusion_BC, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.diffusion = DiffusionModel(state_dim + action_dim, hidden_dim)

    def forward(self, state):
        return self.policy(state)

    def compute_loss(self, state, expert_action):
        pred_action = self.policy(state)
        bc_loss = F.mse_loss(pred_action, expert_action)
        expert_sa = torch.cat([state, expert_action], dim=1)
        diffusion_loss = self.diffusion.compute_loss(expert_sa)
        total_loss = bc_loss + diffusion_loss
        return total_loss, {'bc_loss': bc_loss.item(), 'diffusion_loss': diffusion_loss.item()}

class OT_BC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, ot_lambda=1.0):
        super(OT_BC, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.ot_lambda = ot_lambda

    def forward(self, state):
        return self.policy(state)

    def compute_loss(self, state, expert_action):
        pred_action = self.policy(state)
        bc_loss = F.mse_loss(pred_action, expert_action)
        expert_sa = torch.cat([state, expert_action], dim=1)
        generated_sa = torch.cat([state, pred_action.detach()], dim=1)
        ot_loss_val = ot_loss(expert_sa, generated_sa)
        total_loss = bc_loss + self.ot_lambda * ot_loss_val
        return total_loss, {'bc_loss': bc_loss.item(), 'ot_loss': ot_loss_val}

def train_model(model, dataloader, num_epochs, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    print("Training model: {}".format(model.__class__.__name__))
    for epoch in range(num_epochs):
        epoch_losses = []
        for state, expert_action in dataloader:
            state, expert_action = state.float(), expert_action.float()
            loss, loss_dict = model.compute_loss(state, expert_action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    return model, loss_history
