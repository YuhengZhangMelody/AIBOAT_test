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
    target_side: int = 1
    target_radius: float = 3.0
    transition_radius: float = 6.0
    radius_k: float = 1.0
    danger_safe_radius: float = 3.0
    w_vtpg_far: float = 0.45
    w_vtpg_near: float = 0.10
    w_radius_far: float = 0.05
    w_radius_near: float = 0.30
    w_tangent_far: float = 0.05
    w_tangent_near: float = 0.30
    w_danger: float = 0.25
    w_smooth: float = 0.12
    w_yaw_acc: float = 0.08
    w_step_far: float = 0.03
    w_step_near: float = 0.00
    success_reward: float = 0.8
    collision_penalty: float = -1.0
    out_of_bounds_penalty: float = -0.9
    timeout_penalty: float = -0.5
    reward_clip: float = 1.5

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
            target_side=1,
            target_radius=3.0,
            transition_radius=6.0,
            radius_k=1.0,
            danger_safe_radius=3.0,
            w_vtpg_far=0.45,
            w_vtpg_near=0.10,
            w_radius_far=0.05,
            w_radius_near=0.30,
            w_tangent_far=0.05,
            w_tangent_near=0.30,
            w_danger=0.25,
            w_smooth=0.12,
            w_yaw_acc=0.08,
            w_step_far=0.03,
            w_step_near=0.00,
            success_reward=0.8,
            collision_penalty=-1.0,
            out_of_bounds_penalty=-0.9,
            timeout_penalty=-0.5,
            reward_clip=1.5,
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
            low=np.array(
                [
                    -1000.0, -1000.0, -np.pi, self.env_config.min_action_speed, -10.0, 0.0,
                    -1.0, -1.0, self.env_config.min_action_speed, -10.0, -1.0, -self.env_config.vehicle_max_speed,
                    -self.env_config.vehicle_max_speed, 0.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1000.0, 1000.0, np.pi, self.env_config.vehicle_max_speed, 10.0, 1000.0,
                    1.0, 1.0, self.env_config.vehicle_max_speed, 10.0, 1.0, self.env_config.vehicle_max_speed,
                    self.env_config.vehicle_max_speed, 1000.0,
                ],
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
        self.prev_action = np.array([self.env_config.min_action_speed, 0.0], dtype=np.float32)
        self.target_side = 1 if self.env_config.target_side >= 0 else -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.termination_reason = "running"

        self.boat = Boat(self.start_x, self.start_y, self.start_theta)
        self.boat.v = 1.0
        self.boat.omega = 0.0
        self.waypoint_index = 0
        self.current_step = 0
        self.prev_action = np.array([self.boat.v, self.boat.omega], dtype=np.float32)

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
        cos_err = math.cos(angle_error)
        sin_err = math.sin(angle_error)

        target_buoy = self._get_target_buoy(current_goal)
        d_buoy, e_r, e_t = self._geometry_vectors(target_buoy)
        v_vec = np.array([self.boat.v * math.cos(self.boat.theta), self.boat.v * math.sin(self.boat.theta)])
        v_radial = float(np.dot(v_vec, e_r)) if e_r is not None else 0.0
        v_tangent = float(np.dot(v_vec, e_t)) if e_t is not None else 0.0
        d_buoy_obs = float(d_buoy) if d_buoy is not None else 999.0

        return np.array(
            [
                self.boat.x,
                self.boat.y,
                self.boat.theta,
                self.boat.v,
                self.boat.omega,
                dist_to_goal,
                cos_err,
                sin_err,
                float(self.prev_action[0]),
                float(self.prev_action[1]),
                float(self.target_side),
                v_radial,
                v_tangent,
                d_buoy_obs,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        v, omega = float(clipped_action[0]), float(clipped_action[1])

        pose = np.array([[self.boat.x], [self.boat.y], [self.boat.theta]])
        pose_next = Boat.motion_model_np(pose, [v, omega], self.dt)

        self.boat.x = float(pose_next[0, 0])
        self.boat.y = float(pose_next[1, 0])
        self.boat.theta = float(pose_next[2, 0])
        self.boat.v = v
        self.boat.omega = omega

        reward, terminated, metrics = self._calculate_reward()
        self.current_step += 1

        truncated = False
        if (not terminated) and self.current_step >= self.max_steps:
            truncated = True
            self.termination_reason = "timeout"
            reward += self.env_config.timeout_penalty

        reward = float(np.clip(reward, -self.env_config.reward_clip, self.env_config.reward_clip))
        self.prev_action = np.array([v, omega], dtype=np.float32)

        info = {
            "waypoint_index": self.waypoint_index,
            "termination_reason": self.termination_reason,
            "is_success": self.termination_reason == "success",
            "target_side": self.target_side,
        }
        info.update(metrics)
        return self._get_state(), reward, terminated, truncated, info

    def _get_target_buoy(self, current_goal):
        if not self.buoys:
            return None
        goal_x, goal_y = current_goal.get_position(self.boat)
        return min(self.buoys, key=lambda b: (b.x - goal_x) ** 2 + (b.y - goal_y) ** 2)

    def _geometry_vectors(self, target_buoy):
        if target_buoy is None:
            return None, None, None

        dx = self.boat.x - target_buoy.x
        dy = self.boat.y - target_buoy.y
        d_buoy = math.hypot(dx, dy)
        if d_buoy < 1e-6:
            return d_buoy, np.array([1.0, 0.0]), np.array([0.0, 1.0])

        e_r = np.array([dx / d_buoy, dy / d_buoy], dtype=np.float64)
        if self.target_side > 0:
            e_t = np.array([-e_r[1], e_r[0]], dtype=np.float64)
        else:
            e_t = np.array([e_r[1], -e_r[0]], dtype=np.float64)
        return d_buoy, e_r, e_t

    def _compute_gate_and_weights(self, d_buoy):
        rr = self.env_config.target_radius
        rt = max(self.env_config.transition_radius, rr + 1e-3)
        if d_buoy is None:
            gate = 0.0
        else:
            gate = float(np.clip((rt - d_buoy) / (rt - rr), 0.0, 1.0))

        w_vtpg = self.env_config.w_vtpg_far * (1.0 - gate) + self.env_config.w_vtpg_near * gate
        w_radius = self.env_config.w_radius_far * (1.0 - gate) + self.env_config.w_radius_near * gate
        w_tangent = self.env_config.w_tangent_far * (1.0 - gate) + self.env_config.w_tangent_near * gate
        w_step = self.env_config.w_step_far * (1.0 - gate) + self.env_config.w_step_near * gate
        return gate, w_vtpg, w_radius, w_tangent, w_step

    def _calculate_reward(self):
        terminated = False
        current_goal = self.waypoints[self.waypoint_index]
        goal_x, goal_y = current_goal.get_position(self.boat)
        target_buoy = self._get_target_buoy(current_goal)
        d_buoy, e_r, e_t = self._geometry_vectors(target_buoy)
        _, w_vtpg, w_radius, w_tangent, w_step = self._compute_gate_and_weights(d_buoy)

        v_vec = np.array([self.boat.v * math.cos(self.boat.theta), self.boat.v * math.sin(self.boat.theta)], dtype=np.float64)
        e_goal = np.array([goal_x - self.boat.x, goal_y - self.boat.y], dtype=np.float64)
        goal_norm = np.linalg.norm(e_goal)
        if goal_norm > 1e-6:
            e_goal = e_goal / goal_norm
            r_vtpg = float(np.clip(np.dot(v_vec, e_goal) / max(self.env_config.vehicle_max_speed, 1e-6), -1.0, 1.0))
        else:
            r_vtpg = 0.0

        if d_buoy is not None:
            r_radius = float(np.exp(-self.env_config.radius_k * abs(d_buoy - self.env_config.target_radius)))
            psi_tangent = math.atan2(e_t[1], e_t[0])
            r_tangent = float(math.cos(self.boat.theta - psi_tangent))
            radius_error = abs(d_buoy - self.env_config.target_radius)
        else:
            r_radius = 0.0
            r_tangent = 0.0
            radius_error = float("inf")

        r_danger = 0.0
        for buoy in self.buoys:
            center_dist = math.hypot(self.boat.x - buoy.x, self.boat.y - buoy.y)
            obs_radius = buoy.radius
            safe_radius = max(self.env_config.danger_safe_radius, obs_radius + 1e-3)
            if center_dist <= obs_radius:
                candidate = -1.0
            elif center_dist < safe_radius:
                candidate = -(safe_radius - center_dist) / (safe_radius - obs_radius)
            else:
                candidate = 0.0
            r_danger = min(r_danger, candidate)

        dv_norm = abs(self.boat.v - float(self.prev_action[0])) / max(
            self.env_config.vehicle_max_speed - self.env_config.min_action_speed, 1e-6
        )
        domega_norm = abs(self.boat.omega - float(self.prev_action[1])) / max(2.0 * self.env_config.vehicle_max_yaw_rate, 1e-6)
        r_smooth = -0.5 * (dv_norm + domega_norm)
        r_yaw_acc = -domega_norm
        r_step = -1.0

        reward = (
            w_vtpg * r_vtpg
            + w_radius * r_radius
            + w_tangent * r_tangent
            + self.env_config.w_danger * r_danger
            + self.env_config.w_smooth * r_smooth
            + self.env_config.w_yaw_acc * r_yaw_acc
            + w_step * r_step
        )
        metrics = {
            "r_vtpg": r_vtpg,
            "r_radius": r_radius,
            "r_tangent": r_tangent,
            "r_danger": r_danger,
            "r_smooth": r_smooth,
            "r_yaw_acc": r_yaw_acc,
            "radius_error": float(radius_error if np.isfinite(radius_error) else 999.0),
            "tangent_alignment": float(r_tangent),
            "action_delta": float(0.5 * (dv_norm + domega_norm)),
            "d_buoy": float(d_buoy if d_buoy is not None else 999.0),
            "w_vtpg": float(w_vtpg),
            "w_radius": float(w_radius),
            "w_tangent": float(w_tangent),
            "w_step": float(w_step),
        }

        for buoy in self.buoys:
            if buoy.check_collision_with_boat(self.boat):
                reward += self.env_config.collision_penalty
                self.termination_reason = "collision"
                terminated = True
                return reward, terminated, metrics

        if not self.polygon.is_boat_inside(self.boat):
            reward += self.env_config.out_of_bounds_penalty
            self.termination_reason = "out_of_bounds"
            terminated = True
            return reward, terminated, metrics

        if current_goal.check_collision_with_boat(self.boat, tolerance=3.0):
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                reward += self.env_config.success_reward
                self.termination_reason = "success"
                terminated = True
            else:
                self.previous_distance = self.waypoints[self.waypoint_index].get_distance(self.boat)
        return reward, terminated, metrics

    def render(self):
        pass
