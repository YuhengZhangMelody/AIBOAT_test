from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from boat_env import BoatSlalomEnv

# Créer l'environnement
env = BoatSlalomEnv()

# Vérifier que l'environnement est valide
check_env(env)

# Créer l'agent PPO
model = PPO(
    "MlpPolicy",           # Réseau de neurones simple
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log="./tensorboard_logs/"
)

# Entraîner l'agent
print("Début de l'entraînement...")
model.learn(total_timesteps=1400000)

# Sauvegarder le modèle
model.save("boat_slalom_ppo")
print("Modèle sauvegardé !")
