# Lunar Lander RL Agent Training

Trains an RL agent to safely land a spacecraft between two flags, using Gymnasium's `LunarLander-v3` environment and Stable-Baselines3's PPO algorithm.

## Environment
- **State** (8 values): x/y position, x/y velocity, angle, angular velocity, left/right leg ground contact (boolean).
- **Actions** (4): do nothing, fire left engine, fire main engine, fire right engine.
- **Reward**: shaped — positive for moving toward the pad and reducing speed/tilt, negative for firing engines (fuel cost) and crashing, large terminal bonus (+100) for a safe landing or penalty (-100) for crashing.
- **Solved**: average reward ≥ 200 over 100 consecutive episodes.

## Approach
Same PPO setup style as the CartPole project, but tuned for this harder continuous-dynamics task:
- Higher `gamma=0.999` (longer effective planning horizon — landing depends on decisions many steps earlier)
- `ent_coef=0.01` to encourage exploration (LunarLander has a sparser/trickier reward landscape than CartPole)
- 500,000 timesteps (LunarLander needs substantially more training than CartPole to converge reliably)

## Run
```bash
pip install stable-baselines3[extra] "gymnasium[box2d]" imageio
python train.py
```
**Note**: `gymnasium[box2d]` requires the `box2d-py` package, which needs `swig` installed on your system first (`pip install swig` or `apt install swig` on Linux, `brew install swig` on Mac). Training takes roughly 15-25 minutes on CPU; a GPU won't help much here since PPO with an MLP policy is CPU-bound.

## Outputs
- `training_progress.png` — reward curve over training with the "solved" threshold marked
- `lunarlander_agent.gif` — clip of the trained agent landing
- `lunarlander_ppo_final.zip`, `lunarlander_logs/best_model.zip`

## Expected results
PPO typically **solves LunarLander (reaches ~200+ average reward) somewhere between 300K-500K timesteps**. Early in training you'll usually see the agent learn "don't crash into the ground" before it learns "land gently between the flags" — worth noting in your report as an example of reward shaping guiding curriculum-like learning.

## For your report
Good comparison point with your CartPole project: LunarLander has continuous-valued state dynamics and a much sparser/delayed reward signal for the "successful landing" bonus, which is why it needs ~5-10x more training steps to solve despite using the identical PPO algorithm — a nice illustration of how environment complexity (not just algorithm choice) drives sample efficiency.
