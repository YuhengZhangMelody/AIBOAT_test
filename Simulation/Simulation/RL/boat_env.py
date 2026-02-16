import math
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ajouter le dossier Simulation/Simulation au path de façon robuste
THIS_DIR = Path(__file__).resolve().parent
SIM_DIR = THIS_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from classes import Boat, Buoy, Polygon, Waypoint
from simulation_parameters import parameters


@dataclass
class EnvConfig:
    waypoint_positions: list
    buoy_positions: list
    polygon_points: list
    init_x: float
    init_y: float
    init_yaw: float
    time_step: float
    max_steps: int
    vehicle_max_speed: float
    vehicle_max_yaw_rate: float
    min_action_speed: float = 0.5

    @classmethod
    def from_parameters(cls, params):
        return cls(
            waypoint_positions=list(params.waypoint_positions),
            buoy_positions=list(params.buoy_positions),
            polygon_points=list(params.polygon_points),
            init_x=float(params.init_x),
            init_y=float(params.init_y),
            init_yaw=float(params.init_yaw),
            time_step=float(params.time_step),
            max_steps=2000,
            vehicle_max_speed=float(params.vehicle_max_speed),
            vehicle_max_yaw_rate=float(params.vehicle_max_yaw_rate),
            min_action_speed=0.5,
        )


class BoatSlalomEnv(gym.Env):
    """Gymnasium environment for waypoint slalom."""

    metadata = {"render_modes": []}

    def __init__(self, env_config=None):
        super().__init__()

        self.env_config = env_config or EnvConfig.from_parameters(parameters)
        self.dt = self.env_config.time_step
        self.max_steps = self.env_config.max_steps
        self.current_step = 0
        self.termination_reason = "running"

        self.action_space = spaces.Box(
            low=np.array(
                [self.env_config.min_action_speed, -self.env_config.vehicle_max_yaw_rate],
                dtype=np.float32,
            ),
            high=np.array(
                [self.env_config.vehicle_max_speed, self.env_config.vehicle_max_yaw_rate],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-1000.0, -1000.0, -np.pi, 0.5, -10.0, 0.0, -np.pi], dtype=np.float32),
            high=np.array(
                [1000.0, 1000.0, np.pi, self.env_config.vehicle_max_speed, 10.0, 1000.0, np.pi],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.waypoints = []
        for waypoint in self.env_config.waypoint_positions:
            if isinstance(waypoint, list):
                point1, point2 = waypoint[0], waypoint[1]
                self.waypoints.append(
                    Waypoint(point1[0], point1[1], waypoint_shape="segment", x2=point2[0], y2=point2[1])
                )
            else:
                self.waypoints.append(Waypoint(waypoint[0], waypoint[1]))

        self.buoys = [Buoy(buoy[0], buoy[1]) for buoy in self.env_config.buoy_positions]
        self.polygon = Polygon(self.env_config.polygon_points)

        self.start_x = self.env_config.init_x
        self.start_y = self.env_config.init_y
        self.start_theta = self.env_config.init_yaw

        self.boat = None
        self.waypoint_index = 0
        self.previous_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.termination_reason = "running"

        self.boat = Boat(self.start_x, self.start_y, self.start_theta)
        self.boat.v = 1.0
        self.boat.omega = 0.0
        self.waypoint_index = 0
        self.current_step = 0

        current_goal = self.waypoints[self.waypoint_index]
        self.previous_distance = current_goal.get_distance(self.boat)

        return self._get_state(), {}

    def _get_state(self):
        goal_idx = min(self.waypoint_index, len(self.waypoints) - 1)
        current_goal = self.waypoints[goal_idx]
        dist_to_goal = current_goal.get_distance(self.boat)

        goal_x, goal_y = current_goal.get_position(self.boat)
        angle_to_goal = math.atan2(goal_y - self.boat.y, goal_x - self.boat.x)
        angle_error = angle_to_goal - self.boat.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        return np.array(
            [
                self.boat.x,
                self.boat.y,
                self.boat.theta,
                self.boat.v,
                self.boat.omega,
                dist_to_goal,
                angle_error,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        v, omega = float(action[0]), float(action[1])

        pose = np.array([[self.boat.x], [self.boat.y], [self.boat.theta]])
        pose_next = Boat.motion_model_np(pose, [v, omega], self.dt)

        self.boat.x = float(pose_next[0, 0])
        self.boat.y = float(pose_next[1, 0])
        self.boat.theta = float(pose_next[2, 0])
        self.boat.v = v
        self.boat.omega = omega

        reward, terminated = self._calculate_reward()
        self.current_step += 1

        truncated = False
        if (not terminated) and self.current_step >= self.max_steps:
            truncated = True
            self.termination_reason = "timeout"
            reward -= 50

        info = {
            "waypoint_index": self.waypoint_index,
            "termination_reason": self.termination_reason,
            "is_success": self.termination_reason == "success",
        }
        return self._get_state(), reward, terminated, truncated, info

    def _calculate_reward(self):
        reward = 0.0
        terminated = False
        current_goal = self.waypoints[self.waypoint_index]
        dist_to_goal = current_goal.get_distance(self.boat)

        distance_change = self.previous_distance - dist_to_goal
        if distance_change > 0:
            reward += distance_change * 30.0
        else:
            reward += distance_change * 50.0
        self.previous_distance = dist_to_goal

        goal_x, goal_y = current_goal.get_position(self.boat)
        angle_to_goal = math.atan2(goal_y - self.boat.y, goal_x - self.boat.x)
        angle_error = angle_to_goal - self.boat.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        if abs(angle_error) < math.pi / 2:
            reward += abs(self.boat.v) * 1.0
        else:
            reward -= abs(self.boat.v) * 2.0

        for buoy in self.buoys:
            if buoy.check_collision_with_boat(self.boat):
                reward -= 500
                self.termination_reason = "collision"
                terminated = True
                return reward, terminated

        if not self.polygon.is_boat_inside(self.boat):
            reward -= 500
            self.termination_reason = "out_of_bounds"
            terminated = True
            return reward, terminated

        if current_goal.check_collision_with_boat(self.boat, tolerance=3.0):
            reward += 1000
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                reward += 2000
                self.termination_reason = "success"
                terminated = True
            else:
                self.previous_distance = self.waypoints[self.waypoint_index].get_distance(self.boat)

        reward -= 0.1
        return reward, terminated

    def render(self):
        pass
