#!/usr/bin/env python3
"""
DQN Training Script for JourneyEscape Atari Environment
======================================================

This script trains a Deep Q-Network (DQN) agent to play JourneyEscape using Stable Baselines3.
It supports both MLPPolicy and CNNPolicy, extensive hyperparameter tuning, and comprehensive logging.

Usage:
    python train.py --policy CNN --experiment_name exp1
    python train.py --policy MLP --learning_rate 0.001 --gamma 0.95
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Import and register ALE environments
import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Register ALE environments with gymnasium
try:
    gym.register_envs(ale_py)
except Exception as e:
    print(f"Warning: Could not register ALE environments: {e}")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class HyperparameterExperiment:
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.results_file = os.path.join(base_dir, "hyperparameter_results.csv")
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def log_experiment(self, hyperparams: Dict[str, Any], results: Dict[str, Any]):
        """Log experiment results to CSV file."""
        # Combine hyperparameters and results
        row_data = {**hyperparams, **results}
        row_data["experiment_name"] = self.experiment_name
        row_data["timestamp"] = datetime.now().isoformat()

        # Check if file exists to write header
        file_exists = os.path.exists(self.results_file)

        with open(self.results_file, "a", newline="") as csvfile:
            fieldnames = list(row_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row_data)

        print(f"✅ Experiment results logged to {self.results_file}")


class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "training_timesteps": [],
            "epsilon_values": [],
        }

    def log_episode(self, reward: float, length: int, timestep: int, epsilon: float):
        self.metrics["episode_rewards"].append(reward)
        self.metrics["episode_lengths"].append(length)
        self.metrics["training_timesteps"].append(timestep)
        self.metrics["epsilon_values"].append(epsilon)

    def save_metrics(self, filename: str):
        """Save metrics to file."""
        metrics_df = pd.DataFrame(self.metrics)
        metrics_path = os.path.join(self.log_dir, f"{filename}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        return metrics_path

    def plot_training_curves(self, filename: str):
        if not self.metrics["episode_rewards"]:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        axes[0, 0].plot(self.metrics["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.metrics["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True)

        # Epsilon decay
        axes[1, 0].plot(self.metrics["epsilon_values"])
        axes[1, 0].set_title("Epsilon Decay")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Epsilon")
        axes[1, 0].grid(True)

        # Reward moving average
        if len(self.metrics["episode_rewards"]) >= 10:
            window = min(100, len(self.metrics["episode_rewards"]) // 10)
            moving_avg = (
                pd.Series(self.metrics["episode_rewards"]).rolling(window=window).mean()
            )
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f"Reward Moving Average (window={window})")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Average Reward")
            axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f"{filename}_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return plot_path


def create_environment(env_name: str = "ALE/JourneyEscape-v5", n_envs: int = 1):
    """
    Create and wrap the Atari environment with proper ALE registration.
    """
    print(f"🎮 Creating {env_name} environment...")

    # Ensure ALE is properly registered
    try:
        gym.register_envs(ale_py)
        print("✅ ALE environments registered successfully")
    except Exception as e:
        print(f"⚠️  Warning during ALE registration: {e}")

    # Try alternative environment names if the primary fails
    env_alternatives = [
        env_name,
        "JourneyEscape-v5",
        "JourneyEscape-v4",
        "JourneyEscapeNoFrameskip-v4",
    ]

    env = None
    successful_env_name = None

    for alt_env_name in env_alternatives:
        try:
            print(f"   Trying environment: {alt_env_name}")

            # Create the environment using make_atari_env which handles wrappers
            env = make_atari_env(
                alt_env_name,
                n_envs=n_envs,
                seed=42,
                wrapper_kwargs={
                    "frame_skip": 4,
                    "screen_size": 84,
                    "terminal_on_life_loss": False,
                    "clip_reward": False,
                },
            )

            successful_env_name = alt_env_name
            print(f"✅ Successfully created environment: {alt_env_name}")
            break

        except Exception as e:
            print(f"   ❌ Failed to create {alt_env_name}: {e}")
            continue

    if env is None:
        raise RuntimeError(
            f"Could not create any JourneyEscape environment variant. "
            f"Make sure ALE and Atari ROMs are properly installed."
        )

    # Add frame stacking for temporal information
    env = VecFrameStack(env, n_stack=4)

    print(f"✅ Environment created successfully!")
    print(f"   Environment: {successful_env_name}")
    print(f"   Number of parallel environments: {n_envs}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    return env


def create_dqn_agent(
    env, policy_type: str, hyperparams: Dict[str, Any], tensorboard_log: str
):
    # Policy mapping
    policy_map = {"CNN": "CnnPolicy", "MLP": "MlpPolicy"}

    policy = policy_map.get(policy_type.upper(), "CnnPolicy")

    # DQN configuration
    dqn_config = {
        "policy": policy,
        "env": env,
        "learning_rate": hyperparams.get("learning_rate", 1e-4),
        "gamma": hyperparams.get("gamma", 0.99),
        "batch_size": hyperparams.get("batch_size", 32),
        "buffer_size": hyperparams.get("buffer_size", 100000),
        "learning_starts": hyperparams.get("learning_starts", 10000),
        "target_update_interval": hyperparams.get("target_update_interval", 1000),
        "train_freq": hyperparams.get("train_freq", 4),
        "gradient_steps": hyperparams.get("gradient_steps", 1),
        "exploration_fraction": hyperparams.get("exploration_fraction", 0.1),
        "exploration_initial_eps": hyperparams.get("epsilon_start", 1.0),
        "exploration_final_eps": hyperparams.get("epsilon_end", 0.05),
        "max_grad_norm": hyperparams.get("max_grad_norm", 10),
        "tensorboard_log": tensorboard_log,
        "verbose": 1,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    agent = DQN(**dqn_config)

    return agent, dqn_config


def train_agent(agent, total_timesteps: int, experiment_name: str, save_path: str):
    """Train the DQN agent with callbacks and monitoring."""
    print(" Starting training ...")

    # Setup logging directory
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Configure callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"models/checkpoints_{experiment_name}/",
        name_prefix="dqn_checkpoint",
    )

    # Combine callbacks
    callbacks = CallbackList([checkpoint_callback])

    # Start training
    start_time = time.time()
    agent.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    training_time = time.time() - start_time

    # Save the final model
    agent.save(save_path)
    print(" Model saved to {save_path}")

    return training_time


def evaluate_agent(agent, env, n_eval_episodes: int = 10):
    """Evaluate the trained agent's performance."""
    print(f" Evaluating agent over {n_eval_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Handle vectorized environment returns
            if isinstance(reward, (list, tuple, np.ndarray)):
                episode_reward += reward[0]
                done_val = (
                    done[0] if isinstance(done, (list, tuple, np.ndarray)) else done
                )
                truncated = (
                    info[0].get("TimeLimit.truncated", False)
                    if info and len(info) > 0
                    else False
                )
            else:
                episode_reward += reward
                done_val = done
                truncated = info.get("TimeLimit.truncated", False) if info else False

            episode_length += 1
            done = done_val or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(
            f"   Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}"
        )

    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)

    evaluation_results = {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "avg_episode_length": avg_length,
        "total_eval_episodes": n_eval_episodes,
    }

    print(f"📈 Evaluation Results:")
    print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(
        f"   Min/Max Reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}"
    )
    print(f"   Average Episode Length: {avg_length:.1f}")

    return evaluation_results


def get_predefined_hyperparameter_sets():
    """Return predefined hyperparameter sets for systematic experimentation."""
    return {
        "baseline": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "high_lr": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "low_lr": {
            "learning_rate": 1e-5,
            "gamma": 0.99,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "low_gamma": {
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "high_gamma": {
            "learning_rate": 1e-4,
            "gamma": 0.995,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "large_batch": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "small_batch": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 16,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.1,
        },
        "slow_exploration": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "exploration_fraction": 0.2,
        },
        "fast_exploration": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "exploration_fraction": 0.05,
        },
        "conservative": {
            "learning_rate": 5e-5,
            "gamma": 0.99,
            "batch_size": 64,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "exploration_fraction": 0.15,
        },
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DQN agent for JourneyEscape")
    parser.add_argument(
        "--policy", choices=["CNN", "MLP"], default="CNN", help="Policy type"
    )
    parser.add_argument("--experiment_name", default=None, help="Experiment name")
    parser.add_argument(
        "--hyperparameter_set", default="baseline", help="Predefined hyperparameter set"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=500000, help="Total training timesteps"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10, help="Number of evaluation episodes"
    )

    # Individual hyperparameter overrides
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epsilon_start", type=float, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, help="Final epsilon")
    parser.add_argument(
        "--exploration_fraction", type=float, help="Exploration fraction"
    )

    args = parser.parse_args()

    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"journey_escape_{args.policy.lower()}_{args.hyperparameter_set}_{timestamp}"

    print(f"🚀 Starting DQN Training Experiment: {args.experiment_name}")
    print("=" * 60)

    # Get hyperparameters
    predefined_sets = get_predefined_hyperparameter_sets()
    hyperparams = predefined_sets.get(
        args.hyperparameter_set, predefined_sets["baseline"]
    ).copy()

    # Override with command line arguments
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.gamma is not None:
        hyperparams["gamma"] = args.gamma
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.epsilon_start is not None:
        hyperparams["epsilon_start"] = args.epsilon_start
    if args.epsilon_end is not None:
        hyperparams["epsilon_end"] = args.epsilon_end
    if args.exploration_fraction is not None:
        hyperparams["exploration_fraction"] = args.exploration_fraction

    print("🎛️  Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")
    print()

    # Initialize experiment logger
    experiment_logger = HyperparameterExperiment(args.experiment_name)

    try:
        # Create environment
        env = create_environment()

        # Create agent
        tensorboard_log = f"logs/{args.experiment_name}/tensorboard/"
        agent, agent_config = create_dqn_agent(
            env, args.policy, hyperparams, tensorboard_log
        )

        # Train agent
        model_save_path = f"models/dqn_model_{args.experiment_name}"
        training_time = train_agent(
            agent, args.total_timesteps, args.experiment_name, model_save_path
        )

        # Evaluate agent
        evaluation_results = evaluate_agent(agent, env, args.eval_episodes)

        # Log results
        experiment_results = {
            **evaluation_results,
            "training_time_seconds": training_time,
            "total_timesteps": args.total_timesteps,
            "policy_type": args.policy,
            "hyperparameter_set_name": args.hyperparameter_set,
        }

        experiment_logger.log_experiment(hyperparams, experiment_results)

        print("✅ Training completed successfully!")
        print(f"📁 Model saved as: {model_save_path}.zip")
        print(f"📊 TensorBoard logs: {tensorboard_log}")

        # Save hyperparameter summary
        summary = {
            "experiment_name": args.experiment_name,
            "hyperparameters": hyperparams,
            "results": experiment_results,
            "agent_config": {k: v for k, v in agent_config.items() if k != "env"},
        }

        summary_path = f"experiments/{args.experiment_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📋 Experiment summary saved: {summary_path}")

    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise

    finally:
        env.close()


if __name__ == "__main__":
    main()
