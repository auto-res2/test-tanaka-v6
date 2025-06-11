import torch
import numpy as np
import gym
from preprocess import add_gaussian_noise

class DummyEnv(gym.Env):
    def __init__(self, state_dim=10):
        super(DummyEnv, self).__init__()
        self.state_dim = state_dim
        self.action_dim = 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)
        self.target = np.ones(state_dim) * 0.5
        self.max_steps = 20
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        return (np.random.rand(self.state_dim)*0.1 + self.target).astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        state = np.random.rand(self.state_dim)*0.1 + self.target
        reward = -np.linalg.norm(state - self.target)
        done = (self.current_step >= self.max_steps)
        info = {}
        return state, reward, done, info

def evaluate_policy_ood(model, env, num_episodes=5, noise_std=0.1):
    cumulative_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_noisy = add_gaussian_noise(state_tensor, noise_std)
            with torch.no_grad():
                action = model(state_noisy).squeeze().numpy()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        cumulative_rewards.append(ep_reward)
        print(f"Episode {ep+1} reward: {ep_reward:.4f}")
    avg_reward = np.mean(cumulative_rewards)
    print(f"Average OOD reward: {avg_reward:.4f}")
    return avg_reward
