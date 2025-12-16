import os
import gc
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections import deque
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

GAMMA = 0.99
LR = 1e-3
SOLVED_SCORE = 475
MAX_STEPS = 500


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def to_tensor(state):
    return torch.tensor(state, dtype=torch.float32, device=device)

# Reward-to-go
def compute_returns(rewards):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


# Train
def train():
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)

    policy = PolicyNet().to(device)
    value_fn = ValueNet().to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=LR)
    value_opt = torch.optim.Adam(value_fn.parameters(), lr=LR)

    writer = SummaryWriter()
    reward_buffer = deque(maxlen=100)

    episode = 0
    while True:
        episode += 1
        state, _ = env.reset(seed=seed)

        states, log_probs, rewards = [], [], []
        done = False

        while not done:
            s = to_tensor(state)
            probs = policy(s)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, trunc, _ = env.step(action.item())

            states.append(s)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state
            if trunc:
                break

        # Update
        returns = compute_returns(rewards)
        states = torch.stack(states)

        values = value_fn(states)
        advantages = returns - values.detach()

        # Policy loss
        policy_loss = -(torch.stack(log_probs) * advantages).sum()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        ep_reward = sum(rewards)
        reward_buffer.append(ep_reward)
        avg_reward = np.mean(reward_buffer)

        writer.add_scalar("Reward/Episode", ep_reward, episode)
        writer.add_scalar("Reward/Avg100", avg_reward, episode)

        print(f"Ep {episode} | Reward: {ep_reward} | Avg100: {avg_reward:.1f}")

        if avg_reward >= SOLVED_SCORE and len(reward_buffer) == 100:
            print("CartPole SOLVED!")
            break

    torch.save(policy.state_dict(), "policy.pt")
    writer.close()
    env.close()


# Demo
def demo():
    env = gym.make("CartPole-v1", render_mode="human", max_episode_steps = 3000)
    policy = PolicyNet().to(device)
    policy.load_state_dict(torch.load("policy.pt", map_location=device))
    policy.eval()

    for ep in range(3):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0

        while not done:
            s = to_tensor(state)
            with torch.no_grad():
                action = torch.argmax(policy(s)).item()

            state, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            if trunc:
                break

        print(f"[DEMO] Episode {ep+1} Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    TRAIN_MODE = False  # True = train | False = demo

    if TRAIN_MODE:
        train()
    else:
        demo()
