from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rsaenv import RSAEnv


# ---------- Callback for logging ----------

class EpisodeStatsCallback(BaseCallback):
    """Collect episode returns and blocking rates during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns: List[float] = []
        self.episode_blocking_rates: List[float] = []
        self._current_return: float = 0.0

    def _on_step(self) -> bool:
        # self.locals["rewards"] is a vector because of VecEnv (DummyVecEnv).
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        self._current_return += reward

        if done:
            env: RSAEnv = self.training_env.envs[0]  # unwrap DummyVecEnv
            self.episode_returns.append(self._current_return)
            self._current_return = 0.0
            self.episode_blocking_rates.append(env.get_episode_blocking_rate())

        return True


# ---------- Helpers ----------

def moving_average(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average with padding at the beginning."""
    if len(x) == 0:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad the beginning so lengths match
    pad = np.full(window - 1, ma[0])
    return np.concatenate([pad, ma])


def make_env(request_dir: str, capacity: int, use_random_file: bool) -> RSAEnv:
    """Create a single RSAEnv with all CSVs in the given directory."""
    pattern = os.path.join(request_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No CSV files found under {request_dir!r}")
    return RSAEnv(
        request_files=files,
        capacity=capacity,
        max_holding_time=40,
        use_random_file_per_episode=use_random_file,
    )


def train_dqn_for_capacity(
    capacity: int,
    train_dir: str,
    eval_dir: str,
    total_timesteps: int,
    results_dir: str,
    models_dir: str,
) -> None:
    """Train and evaluate DQN for a given link capacity."""

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Create training environment (randomly pick a training file every episode).
    def _make_train_env():
        return make_env(train_dir, capacity=capacity, use_random_file=True)

    train_env = DummyVecEnv([_make_train_env])

    # 2. Instantiate DQN.
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1_000,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )

    callback = EpisodeStatsCallback()

    # 3. Train.
    print(f"=== Training DQN for capacity={capacity} ===")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 4. Save model.
    model_path = os.path.join(models_dir, f"dqn_capacity_{capacity}.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # 5. Save raw episode stats.
    returns = np.array(callback.episode_returns, dtype=np.float32)
    block_rates = np.array(callback.episode_blocking_rates, dtype=np.float32)
    np.save(os.path.join(results_dir, f"returns_cap{capacity}.npy"), returns)
    np.save(os.path.join(results_dir, f"blocking_cap{capacity}.npy"), block_rates)

    # 6. Plot learning curve (avg return and objective vs episode).
    episodes = np.arange(1, len(returns) + 1)
    ma_returns = moving_average(returns, window=10)
    ma_block = moving_average(block_rates, window=10)

    plt.figure()
    plt.plot(episodes, ma_returns)
    plt.xlabel("Episode")
    plt.ylabel("Avg return (window=10)")
    plt.title(f"DQN learning curve (capacity={capacity})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"learning_curve_cap{capacity}.png"))
    plt.close()

    plt.figure()
    plt.plot(episodes, ma_block)
    plt.xlabel("Episode")
    plt.ylabel("Blocking rate (window=10)")
    plt.title(f"Blocking probability during training (capacity={capacity})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"training_blocking_cap{capacity}.png"))
    plt.close()

    # 7. Evaluation on data/eval with deterministic policy.
    eval_env = make_env(eval_dir, capacity=capacity, use_random_file=False)
    eval_block_rates: List[float] = []

    num_eval_episodes = len(eval_env.request_files)
    print(f"=== Evaluating on {num_eval_episodes} eval files (capacity={capacity}) ===")

    for ep in range(num_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated

        eval_block_rates.append(eval_env.get_episode_blocking_rate())

    eval_block_rates = np.array(eval_block_rates, dtype=np.float32)
    np.save(os.path.join(results_dir, f"eval_blocking_cap{capacity}.npy"), eval_block_rates)

    # 8. Plot evaluation blocking over episodes (each eval episode = one file).
    episodes_eval = np.arange(1, num_eval_episodes + 1)
    ma_eval_block = moving_average(eval_block_rates, window=10)

    plt.figure()
    plt.plot(episodes_eval, ma_eval_block)
    plt.xlabel("Episode (eval file)")
    plt.ylabel("Blocking rate (window=10)")
    plt.title(f"Evaluation blocking on eval set (capacity={capacity})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"eval_blocking_cap{capacity}.png"))
    plt.close()

    print(f"Finished capacity={capacity}. "
          f"Mean eval blocking={eval_block_rates.mean():.4f}")


# ---------- CLI entry point ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--eval_dir", type=str, default="data/eval")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps per capacity.",
    )
    args = parser.parse_args()

    for capacity in (20, 10):
        cap_results = os.path.join(args.results_dir, f"capacity_{capacity}")
        cap_models = os.path.join(args.models_dir, f"capacity_{capacity}")
        train_dqn_for_capacity(
            capacity=capacity,
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            total_timesteps=args.timesteps,
            results_dir=cap_results,
            models_dir=cap_models,
        )


if __name__ == "__main__":
    main()
