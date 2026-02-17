import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from boat_env import BoatSlalomEnv, EnvConfig, parameters


def make_env(env_config, rank, seed=0):
    def _init():
        env = BoatSlalomEnv(env_config=env_config)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for boat slalom.")
    parser.add_argument("--timesteps", type=int, default=1_400_000, help="Total training timesteps.")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel training environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--target-side", type=int, default=1, choices=[-1, 1], help="Orbit direction: +1 CCW, -1 CW.")
    parser.add_argument("--target-radius", type=float, default=3.0, help="Desired orbit radius around buoy (m).")
    parser.add_argument("--transition-radius", type=float, default=6.0, help="Distance where reward weights start transitioning.")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join(script_dir, f"models_{timestamp}")
    logs_dir = os.path.join(script_dir, f"logs_{timestamp}")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env_config = EnvConfig.from_parameters(parameters)
    env_config.target_side = int(args.target_side)
    env_config.target_radius = float(args.target_radius)
    env_config.transition_radius = float(args.transition_radius)

    # Sanity check on a single environment before vectorized training
    check_env(BoatSlalomEnv(env_config=env_config))

    train_env = SubprocVecEnv([make_env(env_config, rank=i, seed=args.seed) for i in range(args.n_envs)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(env_config, rank=10_000, seed=args.seed)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=os.path.join(logs_dir, "eval"),
        eval_freq=max(10_000 // args.n_envs, 1),
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=os.path.join(logs_dir, "tensorboard"),
        seed=args.seed,
    )

    print(f"Model output dir: {models_dir}")
    print(f"Log output dir: {logs_dir}")
    print("Debut de l'entrainement...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    final_model_path = os.path.join(models_dir, "final_model")
    vecnorm_path = os.path.join(models_dir, "vecnormalize.pkl")

    model.save(final_model_path)
    train_env.save(vecnorm_path)

    print(f"Modele final sauvegarde: {final_model_path}.zip")
    print(f"Meilleur modele sauvegarde: {os.path.join(models_dir, 'best_model.zip')}")
    print(f"Statistiques VecNormalize: {vecnorm_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
