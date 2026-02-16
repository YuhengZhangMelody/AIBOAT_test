import argparse
import os
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from boat_env import BoatSlalomEnv, EnvConfig, parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on BoatSlalomEnv.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--model", type=str, default=None, help="Path to PPO zip model.")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize stats (.pkl).")
    parser.add_argument("--target-side", type=int, default=1, choices=[-1, 1], help="Orbit direction: +1 CCW, -1 CW.")
    parser.add_argument("--target-radius", type=float, default=3.0, help="Desired orbit radius around buoy (m).")
    parser.add_argument("--transition-radius", type=float, default=6.0, help="Distance where reward weights start transitioning.")
    return parser.parse_args()


def build_obs_normalizer(vecnorm_path, env_config):
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return None
    dummy_env = DummyVecEnv([lambda: BoatSlalomEnv(env_config=env_config)])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def normalize_state(vecnorm_env, obs):
    if vecnorm_env is None:
        return obs
    normalized = vecnorm_env.normalize_obs(obs.reshape(1, -1))
    return normalized[0]


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_config = EnvConfig.from_parameters(parameters)
    env_config.target_side = int(args.target_side)
    env_config.target_radius = float(args.target_radius)
    env_config.transition_radius = float(args.transition_radius)

    default_model = os.path.join(script_dir, "models", "best_model.zip")
    legacy_model = os.path.join(script_dir, "boat_slalom_ppo_best.zip")
    model_path = args.model or (default_model if os.path.exists(default_model) else legacy_model)

    default_vecnorm = os.path.join(script_dir, "models", "vecnormalize.pkl")
    vecnorm_path = args.vecnorm or default_vecnorm

    model = PPO.load(model_path)
    obs_normalizer = build_obs_normalizer(vecnorm_path, env_config)

    env = BoatSlalomEnv(env_config=env_config)
    reasons = Counter()
    episode_returns = []
    episode_lengths = []
    action_deltas = []
    radius_errors = []
    tangent_alignments = []

    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        done = False
        total_reward = 0.0
        steps = 0
        final_reason = "unknown"

        while not done:
            policy_obs = normalize_state(obs_normalizer, obs)
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if done:
                final_reason = info.get("termination_reason", "unknown")
            action_deltas.append(float(info.get("action_delta", 0.0)))
            radius_error = float(info.get("radius_error", 999.0))
            if radius_error < 900:
                radius_errors.append(radius_error)
            tangent_alignments.append(float(info.get("tangent_alignment", 0.0)))

        reasons[final_reason] += 1
        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    success = reasons.get("success", 0)
    print("Evaluation summary")
    print(f"Episodes: {args.episodes}")
    print(f"Model: {model_path}")
    if obs_normalizer is not None:
        print(f"VecNormalize: {vecnorm_path}")
    else:
        print("VecNormalize: not used")
    print(f"Success rate: {success / args.episodes:.2%}")
    print(f"Collision rate: {reasons.get('collision', 0) / args.episodes:.2%}")
    print(f"Out-of-bounds rate: {reasons.get('out_of_bounds', 0) / args.episodes:.2%}")
    print(f"Timeout rate: {reasons.get('timeout', 0) / args.episodes:.2%}")
    print(f"Mean return: {np.mean(episode_returns):.3f} +/- {np.std(episode_returns):.3f}")
    print(f"Mean steps: {np.mean(episode_lengths):.2f}")
    if action_deltas:
        print(f"Mean action delta: {np.mean(action_deltas):.4f}")
    if radius_errors:
        print(f"Mean radius error: {np.mean(radius_errors):.4f} m")
    if tangent_alignments:
        print(f"Mean tangent alignment: {np.mean(tangent_alignments):.4f}")
    print(f"Termination breakdown: {dict(reasons)}")

    env.close()
    if obs_normalizer is not None:
        obs_normalizer.close()


if __name__ == "__main__":
    main()
