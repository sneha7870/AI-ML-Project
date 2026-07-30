"""
Lunar Lander RL Agent Training
----------------------------------
Trains an RL agent to land a spacecraft safely between two flags using
Gymnasium's LunarLander-v3 environment and Stable-Baselines3 (PPO).

LunarLander-v2/v3: agent controls main + side thrusters to land softly.
Reward shaped by: proximity to landing pad, speed, angle, leg contact,
fuel usage, and a large terminal bonus/penalty for landing/crashing.
"Solved" = average reward >= 200 over 100 consecutive episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

LOG_DIR = "lunarlander_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. SETUP ENVIRONMENT
# ---------------------------------------------------------------------------
# Requires: pip install "gymnasium[box2d]"
env = Monitor(gym.make("LunarLander-v3"), LOG_DIR)
eval_env = Monitor(gym.make("LunarLander-v3"))

print(f"Observation space: {env.observation_space}")
# [x, y, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]
print(f"Action space: {env.action_space}")
# 0=do nothing, 1=fire left engine, 2=fire main engine, 3=fire right engine

# ---------------------------------------------------------------------------
# 2. TRAIN WITH PPO
# ---------------------------------------------------------------------------
eval_callback = EvalCallback(
    eval_env, best_model_save_path=LOG_DIR,
    log_path=LOG_DIR, eval_freq=5000,
    deterministic=True, render=False,
    n_eval_episodes=10,
)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.98,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

TOTAL_TIMESTEPS = 500_000  # LunarLander needs more steps than CartPole to converge well
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
model.save("lunarlander_ppo_final")

# ---------------------------------------------------------------------------
# 3. EVALUATE TRAINED AGENT
# ---------------------------------------------------------------------------
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f"\nFinal evaluation over 100 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Solved (>=200 avg reward)? {'YES' if mean_reward >= 200 else 'NO'}")

# ---------------------------------------------------------------------------
# 4. PLOT TRAINING PROGRESS
# ---------------------------------------------------------------------------
x, y = ts2xy(load_results(LOG_DIR), "timesteps")
if len(x) > 0:
    window = 20
    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid") if len(y) >= window else y
    x_smooth = x[len(x) - len(y_smooth):]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.25, label="raw episode reward")
    plt.plot(x_smooth, y_smooth, label=f"{window}-episode moving avg", linewidth=2)
    plt.axhline(200, color="red", linestyle="--", label="solved threshold")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("LunarLander-v3 PPO Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    plt.close()
    print("Saved training_progress.png")

# ---------------------------------------------------------------------------
# 5. RECORD A GIF OF THE TRAINED AGENT
# ---------------------------------------------------------------------------
try:
    import imageio
    render_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames = []
    obs, _ = render_env.reset()
    for _ in range(600):
        frames.append(render_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        if terminated or truncated:
            obs, _ = render_env.reset()
    imageio.mimsave("lunarlander_agent.gif", frames, fps=30)
    print("Saved lunarlander_agent.gif")
except ImportError:
    print("imageio not installed — skipping GIF export. `pip install imageio` to enable it.")
except Exception as e:
    print(f"Rendering skipped ({e}) — likely no display/box2d available; fine for headless training.")

print("Done.")
