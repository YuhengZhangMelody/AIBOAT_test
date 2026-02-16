# boat_env.py (VERSION FINALE CORRIGÉE)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, '..')

from classes import Boat, Waypoint, Buoy, Polygon
from simulation_parameters import parameters


class BoatSlalomEnv(gym.Env):
    """Environnement Gym pour slalom entre 2 bouées"""
    
    def __init__(self):
        super(BoatSlalomEnv, self).__init__()
        
        # Définir l'espace d'action : [v, omega]
        # Vitesse minimum = 0.5 pour forcer le mouvement
        self.action_space = spaces.Box(
            low=np.array([0.5, -parameters.vehicle_max_yaw_rate], dtype=np.float32),
            high=np.array([parameters.vehicle_max_speed, parameters.vehicle_max_yaw_rate], dtype=np.float32),
            dtype=np.float32
        )
        
        # Définir l'espace d'observation : [x, y, theta, v, omega, dist_to_goal, angle_error]
        self.observation_space = spaces.Box(
            low=np.array([-1000.0, -1000.0, -np.pi, 0.5, -10.0, 0.0, -np.pi], dtype=np.float32),
            high=np.array([1000.0, 1000.0, np.pi, parameters.vehicle_max_speed, 10.0, 1000.0, np.pi], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialisation
        self.dt = parameters.time_step
        self.max_steps = 2000
        self.current_step = 0
        
        # Créer les waypoints pour le slalom
        self.waypoints = []
        for waypoint in parameters.waypoint_positions:
            if isinstance(waypoint, list):
                point1, point2 = waypoint[0], waypoint[1]
                self.waypoints.append(Waypoint(point1[0], point1[1], waypoint_shape='segment', x2=point2[0], y2=point2[1]))
            else:
                self.waypoints.append(Waypoint(waypoint[0], waypoint[1]))
        
        # Créer les obstacles (bouées)
        self.buoys = [Buoy(buoy[0], buoy[1]) for buoy in parameters.buoy_positions]
        
        # Polygone de zone de travail
        self.polygon = Polygon(parameters.polygon_points)
        
        # Position de départ
        self.start_x = parameters.init_x
        self.start_y = parameters.init_y
        self.start_theta = parameters.init_yaw
        
        # Initialiser le bateau
        self.boat = None
        self.waypoint_index = 0
        self.previous_distance = 0
        
    def reset(self, seed=None, options=None):
        """Réinitialiser l'environnement"""
        super().reset(seed=seed)
        
        self.boat = Boat(self.start_x, self.start_y, self.start_theta)
        self.boat.v = 1.0  # Commence avec vitesse
        self.boat.omega = 0.0
        self.waypoint_index = 0
        self.current_step = 0
        
        current_goal = self.waypoints[self.waypoint_index]
        self.previous_distance = current_goal.get_distance(self.boat)
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Retourner l'état actuel avec angle vers le goal"""
        current_goal = self.waypoints[self.waypoint_index]
        dist_to_goal = current_goal.get_distance(self.boat)
        
        # Calculer l'angle vers le goal
        goal_x, goal_y = current_goal.get_position(self.boat)
        angle_to_goal = math.atan2(goal_y - self.boat.y, goal_x - self.boat.x)
        angle_error = angle_to_goal - self.boat.theta
        # Normaliser entre -pi et pi
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        return np.array([
            self.boat.x,
            self.boat.y,
            self.boat.theta,
            self.boat.v,
            self.boat.omega,
            dist_to_goal,
            angle_error  # NOUVEAU !
        ], dtype=np.float32)
    
    def step(self, action):
        """Exécuter une action"""
        v, omega = float(action[0]), float(action[1])
        
        # Mettre à jour le bateau
        pose = np.array([[self.boat.x], [self.boat.y], [self.boat.theta]])
        pose_next = Boat.motion_model_np(pose, [v, omega], self.dt)
        
        self.boat.x = float(pose_next[0, 0])
        self.boat.y = float(pose_next[1, 0])
        self.boat.theta = float(pose_next[2, 0])
        self.boat.v = v
        self.boat.omega = omega
        
        # Calculer la reward
        reward, terminated = self._calculate_reward()
        
        self.current_step += 1
        
        # Timeout
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 50
        
        return self._get_state(), reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        reward = 0.0
        terminated = False
        
        current_goal = self.waypoints[self.waypoint_index]
        dist_to_goal = current_goal.get_distance(self.boat)
        
        # 1. Récompense/Pénalité pour rapprochement/éloignement
        distance_change = self.previous_distance - dist_to_goal
        
        if distance_change > 0:  # Se rapproche
            reward += distance_change * 30.0  # GROSSE récompense
        else:  # S'éloigne
            reward += distance_change * 50.0  # GROSSE pénalité (sera négatif)
        
        self.previous_distance = dist_to_goal
        
        # 2. Récompense pour avancer UNIQUEMENT si bien orienté
        goal_x, goal_y = current_goal.get_position(self.boat)
        angle_to_goal = math.atan2(goal_y - self.boat.y, goal_x - self.boat.x)
        angle_error = angle_to_goal - self.boat.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        # Récompense vitesse SEULEMENT si orienté vers le goal
        if abs(angle_error) < math.pi / 2:  # Orienté dans la bonne direction (±90°)
            reward += abs(self.boat.v) * 1.0
        else:  # Orienté dans la mauvaise direction
            reward -= abs(self.boat.v) * 2.0  # Pénalité pour aller dans le mauvais sens
        
        # 3. Pénalité collision avec obstacles
        for buoy in self.buoys:
            if buoy.check_collision_with_boat(self.boat):
                reward -= 500
                terminated = True
                return reward, terminated
        
        # 4. Pénalité sortie du polygone
        if not self.polygon.is_boat_inside(self.boat):
            reward -= 500
            terminated = True
            return reward, terminated
        
        # 5. Récompense atteinte du waypoint
        if current_goal.check_collision_with_boat(self.boat, tolerance=3.0):
            reward += 1000  # GROSSE récompense
            self.waypoint_index += 1
            
            if self.waypoint_index >= len(self.waypoints):
                reward += 2000  # ÉNORME récompense finale
                terminated = True
            else:
                self.previous_distance = self.waypoints[self.waypoint_index].get_distance(self.boat)
        
        # 6. Petite pénalité temps
        reward -= 0.1
        
        return reward, terminated
    
    def render(self):
        """Affichage (optionnel)"""
        pass
