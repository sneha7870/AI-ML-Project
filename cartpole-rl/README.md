# CartPole RL Agent Training

Trains a reinforcement learning agent to balance a pole on a moving cart, using Gymnasium's `CartPole-v1` environment and Stable-Baselines3's PPO algorithm.

## Environment
- **State** (4 values): cart position, cart velocity, pole angle, pole angular velocity.
- **Actions** (2): push cart left or right.
- **Reward**: +1 per timestep the pole stays upright (max 500 per episode).
- **Solved**: average reward ≥ 475 over 100 consecutive episodes.

## Approach
- **Algorithm**: PPO (Proximal Policy Optimization) — a stable, on-policy actor-critic method that's the go-to baseline for classic control tasks like this.
- **Training**: 100,000 timesteps, with periodic evaluation callbacks that save the best-performing checkpoint.
- A **from-scratch PyTorch DQN implementation is included as a commented appendix** at the bottom of `train.py` — uncomment `train_dqn_from_scratch()` if your assignment specifically wants you to show the underlying algorithm (replay buffer, target network, epsilon-greedy) rather than a library call.

## Run
```bash
pip install stable-baselines3[extra] gymnasium imageio
python train.py
```
Trains in a few minutes on CPU — CartPole is lightweight, no GPU needed.

## Outputs
- `training_progress.png` — reward per episode over training (with 20-episode moving average and the "solved" threshold marked)
- `cartpole_agent.gif` — a short clip of the trained agent balancing the pole (only generated if a display/renderer is available; skipped gracefully in headless environments)
- `cartpole_ppo_final.zip`, `cartpole_logs/best_model.zip`

## Expected results
PPO typically **solves CartPole (reaches ~500 reward) within 30,000-60,000 timesteps** — you should see the moving average curve flatten near the max reward well before the full 100K steps complete.

## For your report
Good things to discuss: why PPO's clipped objective avoids destructively large policy updates, the exploration/exploitation tradeoff (contrast with DQN's epsilon-greedy if you run the appendix version), and how reward shaping isn't needed here since CartPole's dense +1-per-step reward already gives a clean learning signal.
