"""
CartPole RL Agent Training
------------------------------
Trains a reinforcement learning agent to balance a pole on a cart using
Gymnasium's CartPole-v1 environment. Uses Stable-Baselines3 (PPO), with
a from-scratch DQN implementation included for comparison / learning purposes.

CartPole-v1: agent gets +1 reward per timestep the pole stays upright.
Episode ends if pole falls >15deg or cart moves off screen. "Solved" =
average reward >= 475 over 100 consecutive episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

import os

LOG_DIR = "cartpole_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. SETUP ENVIRONMENT
# ---------------------------------------------------------------------------
env = Monitor(gym.make("CartPole-v1"), LOG_DIR)
eval_env = Monitor(gym.make("CartPole-v1"))

print(f"Observation space: {env.observation_space}")  # [cart pos, cart vel, pole angle, pole ang vel]
print(f"Action space: {env.action_space}")             # 0 = push left, 1 = push right

# ---------------------------------------------------------------------------
# 2. TRAIN WITH PPO (recommended — stable, sample-efficient for CartPole)
# ---------------------------------------------------------------------------
eval_callback = EvalCallback(
    eval_env, best_model_save_path=LOG_DIR,
    log_path=LOG_DIR, eval_freq=2000,
    deterministic=True, render=False,
    n_eval_episodes=10,
)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

TOTAL_TIMESTEPS = 100_000
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
model.save("cartpole_ppo_final")

# ---------------------------------------------------------------------------
# 3. EVALUATE TRAINED AGENT
# ---------------------------------------------------------------------------
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f"\nFinal evaluation over 100 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Solved (>=475 avg reward)? {'YES' if mean_reward >= 475 else 'NO'}")

# ---------------------------------------------------------------------------
# 4. PLOT TRAINING PROGRESS (reward per episode over training)
# ---------------------------------------------------------------------------
x, y = ts2xy(load_results(LOG_DIR), "timesteps")
if len(x) > 0:
    # smooth with a rolling average
    window = 20
    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid") if len(y) >= window else y
    x_smooth = x[len(x) - len(y_smooth):]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.3, label="raw episode reward")
    plt.plot(x_smooth, y_smooth, label=f"{window}-episode moving avg", linewidth=2)
    plt.axhline(475, color="red", linestyle="--", label="solved threshold")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("CartPole-v1 PPO Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    plt.close()
    print("Saved training_progress.png")

# ---------------------------------------------------------------------------
# 5. RECORD A GIF OF THE TRAINED AGENT (optional, needs imageio)
# ---------------------------------------------------------------------------
try:
    import imageio
    frames = []
    obs, _ = eval_env.reset()
    for _ in range(500):
        frames.append(eval_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            obs, _ = eval_env.reset()
    imageio.mimsave("cartpole_agent.gif", frames, fps=30)
    print("Saved cartpole_agent.gif")
except ImportError:
    print("imageio not installed — skipping GIF export. `pip install imageio` to enable it.")
except Exception as e:
    print(f"Rendering skipped ({e}) — likely no display available; this is fine for headless training.")

print("Done.")

# ---------------------------------------------------------------------------
# APPENDIX: from-scratch DQN (for comparison / understanding fundamentals)
# ---------------------------------------------------------------------------
# Uncomment to also train a simple DQN from scratch with PyTorch instead of
# using Stable-Baselines3's implementation. Useful if your assignment wants
# you to show the underlying algorithm rather than a library call.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(args)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buffer)

def train_dqn_from_scratch(episodes=500, gamma=0.99, lr=1e-3, batch_size=64,
                            eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = eps_start
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(state)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s = torch.FloatTensor(s); s2 = torch.FloatTensor(s2)
                a = torch.LongTensor(a); r = torch.FloatTensor(r); d = torch.FloatTensor(d)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    next_q = target_net(s2).max(1)[0]
                    target = r + gamma * next_q * (1 - d)

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        epsilon = max(eps_end, epsilon * eps_decay)
        rewards_history.append(total_reward)

        if ep % 20 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    return q_net, rewards_history

# q_net, rewards_history = train_dqn_from_scratch()
"""
