# main_rl.py (version VRAIMENT corrigée)
import pygame
import numpy as np
import math
import sys
import os
import argparse
"ALLER dans IABoat/Simulation/Simulation/simulation_parameters.py "
"et tout commenter sauf parameters.test_waypoint_segments()"

# Obtenir le chemin absolu du dossier où se trouve ce script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Aller au dossier parent (Simulation/Simulation)
parent_dir = os.path.dirname(script_dir)


# Aller au grand-parent (Simulation) - where Images folder is
grandparent_dir = os.path.dirname(parent_dir)


# Changer le working directory to where Images folder exists
os.chdir(grandparent_dir)


# Ajouter les dossiers nécessaires au sys.path
sys.path.insert(0, parent_dir)  # for simulation modules
sys.path.insert(0, os.path.join(grandparent_dir, "Pythonrobotics"))


from simulation_parameters import parameters
from pygame_manager import *
from classes import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from boat_env import BoatSlalomEnv, EnvConfig



def get_rl_state(boat, current_goal, buoys, prev_action, target_side):
    """Build policy observation consistent with BoatSlalomEnv."""
    dist_to_goal = current_goal.get_distance(boat)

    goal_x, goal_y = current_goal.get_position(boat)
    angle_to_goal = math.atan2(goal_y - boat.y, goal_x - boat.x)
    angle_error = angle_to_goal - boat.theta
    cos_err = math.cos(angle_error)
    sin_err = math.sin(angle_error)

    target_buoy = None
    if buoys:
        target_buoy = min(buoys, key=lambda b: (b.x - goal_x) ** 2 + (b.y - goal_y) ** 2)

    v_vec = np.array([boat.v * math.cos(boat.theta), boat.v * math.sin(boat.theta)], dtype=np.float64)
    v_radial = 0.0
    v_tangent = 0.0
    d_buoy = 999.0

    if target_buoy is not None:
        dx = boat.x - target_buoy.x
        dy = boat.y - target_buoy.y
        d_buoy = math.hypot(dx, dy)
        if d_buoy < 1e-6:
            e_r = np.array([1.0, 0.0], dtype=np.float64)
        else:
            e_r = np.array([dx / d_buoy, dy / d_buoy], dtype=np.float64)

        if target_side > 0:
            e_t = np.array([-e_r[1], e_r[0]], dtype=np.float64)
        else:
            e_t = np.array([e_r[1], -e_r[0]], dtype=np.float64)

        v_radial = float(np.dot(v_vec, e_r))
        v_tangent = float(np.dot(v_vec, e_t))

    return np.array(
        [
            boat.x,
            boat.y,
            boat.theta,
            boat.v,
            boat.omega,
            dist_to_goal,
            cos_err,
            sin_err,
            float(prev_action[0]),
            float(prev_action[1]),
            float(target_side),
            v_radial,
            v_tangent,
            d_buoy,
        ],
        dtype=np.float32,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO policy with Pygame visualization.")
    parser.add_argument("--model", type=str, default=None, help="Path to PPO zip model.")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize stats (.pkl).")
    parser.add_argument("--target-side", type=int, default=1, choices=[-1, 1], help="Orbit direction: +1 CCW, -1 CW.")
    return parser.parse_args()


def build_obs_normalizer(vecnorm_path, target_side):
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return None

    env_config = EnvConfig.from_parameters(parameters)
    env_config.target_side = int(target_side)
    dummy_env = DummyVecEnv([lambda: BoatSlalomEnv(env_config=env_config)])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def normalize_state(vecnorm_env, state):
    if vecnorm_env is None:
        return state
    normalized = vecnorm_env.normalize_obs(state.reshape(1, -1))
    return normalized[0]


def main():
    """Main simulation function with RL agent"""
    args = parse_args()
    print(__file__ + " start!!")

    #################### Initialisation ####################


    ####### Elements de la simulation #######
    # Load Waypoints
    waypoints = []
    number = 1
    for waypoint in parameters.waypoint_positions:
        if isinstance(waypoint, list):
            point1 = waypoint[0]
            point2 = waypoint[1]
            waypoints.append(Waypoint(point1[0], point1[1], waypoint_type='goal', number=number, waypoint_shape='segment', x2=point2[0], y2=point2[1]))
        else:
            waypoint_pos_x, waypoint_pos_y = waypoint
            waypoints.append(Waypoint(waypoint_pos_x, waypoint_pos_y, waypoint_type='goal', number=number))
        number += 1
    
    waypoints_nb = 0
    current_goal = waypoints[waypoints_nb]


    # Load Buoys
    buoys = []
    for buoy in parameters.buoy_positions:
        buoys.append(Buoy(buoy[0], buoy[1]))


    # Polygone de zone de travail
    polygon = Polygon(parameters.polygon_points)


    # Modèle de bateau
    boat = Boat(parameters.init_x, parameters.init_y, parameters.init_yaw, alpha=150)
    boat.v = max(0.5, parameters.init_v)
    boat.omega = 0.0
    prev_action = np.array([boat.v, boat.omega], dtype=np.float32)
    target_side = int(args.target_side)


    ####### RL Agent #######
    default_model = os.path.join(script_dir, "models", "best_model.zip")
    legacy_model = os.path.join(script_dir, "boat_slalom_ppo_best.zip")
    model_path = args.model or (default_model if os.path.exists(default_model) else legacy_model)

    default_vecnorm = os.path.join(script_dir, "models", "vecnormalize.pkl")
    vecnorm_path = args.vecnorm or default_vecnorm

    try:
        model = PPO.load(model_path)
        print("Modèle RL chargé avec succès!")
    except FileNotFoundError:
        print(f"ERREUR: Modèle '{model_path}' non trouvé!")
        print("Entraîne d'abord le modèle avec train_rl.py")
        return

    obs_normalizer = build_obs_normalizer(vecnorm_path, args.target_side)
    if obs_normalizer is None:
        print("Info: aucune normalisation d'observation chargée (VecNormalize absent).")
    else:
        print(f"Statistiques VecNormalize chargées: {vecnorm_path}")


    # No initial command
    u = [0, 0]


    ####### AUTRES #######
    trajectory = []
    time = 0.0
    dt = parameters.time_step


    #################### Main loop ####################
    running = True
    final_goal_reached = False


    while running and not final_goal_reached:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False


        ### --------- CALCUL --------- ####


        ####### RL Agent #######
        # Obtenir l'état pour l'agent RL
        state = get_rl_state(boat, current_goal, buoys, prev_action, target_side)
        state_for_policy = normalize_state(obs_normalizer, state)
        
        # L'agent RL choisit l'action [v, omega]
        action, _ = model.predict(state_for_policy, deterministic=True)
        u = action.tolist()
        
        # Update robot state directement (pas de SLAM)
        pose = np.array([[boat.x], [boat.y], [boat.theta]])
        pose_next = Boat.motion_model_np(pose, u, dt)
        
        boat.x = pose_next[0, 0]
        boat.y = pose_next[1, 0]
        boat.theta = pose_next[2, 0]
        boat.v = u[0]
        boat.omega = u[1]
        prev_action = np.array([boat.v, boat.omega], dtype=np.float32)
        
        # Store trajectory
        trajectory.append([boat.x, boat.y, boat.theta, boat.v, boat.omega])
        
        # Maintain trail
        boat.trail.append((int(boat.x), int(boat.y)))
        if len(boat.trail) > boat.max_trail_length:
            boat.trail.pop(0)


        ### --------- AFFICHAGE --------- ####
        if parameters.show_animation:
            screen.fill(parameters.BLUE)
            draw_basic_screen(screen, boat.x, boat.y, boat.theta)
            
            for buoy in buoys:
                buoy.draw(screen, boat.x, boat.y, boat.theta)
            
            for waypoint in waypoints:
                waypoint.draw(screen, font, boat.x, boat.y, boat.theta)
            
            # ✨ AJOUT : Dessiner la trajectoire en temps réel ✨
            if len(trajectory) > 1:
                trajectory_array = np.array(trajectory)
                temp_x = trajectory_array[:, 0]
                temp_y = trajectory_array[:, 1]
                draw_reference_path(screen, temp_x, temp_y, boat.x, boat.y, boat.theta)
            
            # Draw boat
            boat.show(screen)
            boat.display_state(screen, time, font)
            boat.draw_arrow(screen)
            
            # Draw polygon
            polygon.draw(screen, boat.x, boat.y, boat.theta)
            
            # Afficher "RL Agent"
            rl_text = font.render("RL Agent", True, parameters.BLACK)
            screen.blit(rl_text, (10, 130))
            
            pygame.display.flip()
            clock.tick(60)


        # Check if goal is reached
        if current_goal.check_collision_with_boat(boat, tolerance=2.0):
            current_goal.reached = True
            waypoints_nb += 1
            print("Goal!!")
            
            if waypoints_nb >= len(waypoints):
                final_goal_reached = True
            else:
                current_goal = waypoints[waypoints_nb]


        # Update time
        time += dt


    print("Done")


    # Display final trajectory
    if parameters.show_animation and final_goal_reached:
        screen.fill(parameters.WHITE)
        draw_basic_screen(screen, boat.x, boat.y, boat.theta)
        
        for buoy in buoys:
            buoy.draw(screen, boat.x, boat.y, boat.theta)
        
        for waypoint in waypoints:
            waypoint.draw(screen, font, boat.x, boat.y, boat.theta)
        
        # Draw final trajectory
        trajectory_array = np.array(trajectory)
        temp_x = trajectory_array[:, 0]
        temp_y = trajectory_array[:, 1]
        draw_reference_path(screen, temp_x, temp_y, boat.x, boat.y, boat.theta)
        
        polygon.draw(screen, boat.x, boat.y, boat.theta)
        
        pygame.display.flip()


        # Wait for user to close window
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False


    pygame.quit()
    if obs_normalizer is not None:
        obs_normalizer.close()



if __name__ == '__main__':
    main()
