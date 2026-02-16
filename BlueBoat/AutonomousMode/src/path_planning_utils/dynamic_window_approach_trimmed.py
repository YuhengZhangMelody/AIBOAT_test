"""

Mobile robot motion planning sample with Dynamic Window Approach

This module implements the Dynamic Window Approach (DWA) for mobile robot navigation.
DWA is a local motion planning algorithm that evaluates multiple trajectories in a
velocity space (dynamic window) to select the optimal control inputs that avoid
obstacles while heading towards the goal.

The algorithm works by:
1. Computing the dynamic window based on robot constraints and current state
2. Sampling velocity commands within this window
3. Predicting trajectories for each sample
4. Evaluating trajectories using cost functions (goal distance, speed, obstacle avoidance)
5. Selecting the trajectory with minimum cost

Visualization is provided using Pygame for real-time animation.

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

from itertools import count
import math
import pygame
import numpy as np
# IMPORTANT LA CONFIG EST FAITE A PARTIR DE CE FICHIER IMPORTE
from src.path_planning_utils.simulation_parameters import parameters
from src.path_planning_utils.classes import Boat, Buoy, Polygon


def dwa_control(x, config, goals, ob, current_goal_index):
    """
    Dynamic Window Approach control

    Computes the optimal velocity commands (linear and angular) by evaluating
    trajectories within the dynamic window and selecting the one with minimum cost.

    Args:
        x: Current robot state [x, y, yaw, v, omega]
        config: Configuration object with simulation parameters
        goal: Waypoint object representing the target
        ob: Obstacle positions array

    Returns:
        u: Optimal control inputs [v, omega]
        trajectory: Predicted trajectory for the optimal inputs
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goals, ob, current_goal_index)

    return u, trajectory

class Config:
    """
    Simulation parameter class containing all configuration parameters for DWA.

    This class holds robot physical constraints, cost function weights, obstacle
    positions, and visualization settings used throughout the simulation.
    """

    def __init__(self, params=None):
        """
        Initialize Config with parameters from simulation_parameters.
        
        Args:
            params: Optional Parameters instance. If None, uses the global parameters instance.
        """
        if params is None:
            params = parameters
            
        # Robot kinematic constraints
        self.max_speed = params.vehicle_max_speed  # [m/s] Maximum linear velocity
        self.min_speed = params.vehicle_min_speed  # [m/s] Minimum linear velocity (allows backward motion)
        self.max_yaw_rate = params.vehicle_max_yaw_rate  # [rad/s] Maximum angular velocity
        self.max_accel = params.vehicle_max_accel  # [m/ss] Maximum linear acceleration
        self.max_delta_yaw_rate = params.vehicle_max_delta_yaw_rate  # [rad/ss] Maximum angular acceleration

        # Sampling
        self.v_resolution = params.sampling_v_resolution  # [m/s] Resolution for velocity sampling
        self.yaw_rate_resolution = params.sampling_yaw_rate_resolution  # [rad/s] Resolution for yaw rate sampling
        self.dt = params.time_step # [s] Time step for motion prediction
        self.predict_time = params.time_horizon # [s] Prediction horizon for trajectory evaluation

        # Cost function weights
        self.to_goal_cost_gain = params.to_goal_cost_gain # Weight for goal-oriented cost
        self.speed_cost_gain = params.speed_cost_gain  # Weight for preferring higher speeds
        self.obstacle_cost_gain = params.obstacle_cost_gain  # Weight for obstacle avoidance
        self.close_cost_gain = params.close_cost_gain # Weight for goal closeness
        self.max_dist_for_cost = params.max_dist_for_cost # [m] distance maximale après laquelle le cout est le même
        self.robot_stuck_flag_cons = 0.001  # [m/s] Threshold to detect robot stuck condition
        self.infinite_cost = float('inf')  # Cost value representing an infeasible trajectory
        
        # Work area polygon
        self.polygon = Polygon(params.polygon_points)

config = Config()


def motion(x, u, dt):
    """
    Motion model for robot kinematics using Boat.motion_model_np.

    Updates the robot state based on velocity commands.
    Input state vector: [x, y, yaw, v, omega]
    Output state vector: [x, y, yaw, v, omega] (with v, omega updated from control u)

    Args:
        x: Current state [x, y, yaw, v, omega]
        u: Control inputs [linear_velocity, angular_velocity]
        dt: Time step

    Returns:
        Updated state vector
    """
    # Extract pose [x, y, yaw] as (3,1) numpy array
    pose = np.array([[x[0]], [x[1]], [x[2]]])

    # Apply Boat motion model
    pose_next = Boat.motion_model_np(pose, u, dt)

    # Update state vector with new pose and velocities
    x[0] = pose_next[0, 0]
    x[1] = pose_next[1, 0]
    x[2] = pose_next[2, 0]
    x[3] = u[0]  # Update linear velocity
    x[4] = u[1]  # Update angular velocity

    return x


def calc_dynamic_window(x, config):
    """
    Calculate the dynamic window based on current state and constraints.

    The dynamic window is the set of achievable velocities in the next time step,
    constrained by robot physical limits and current motion state.

    Args:
        x: Current robot state [x, y, yaw, v, omega]
        config: Configuration parameters

    Returns:
        dw: Dynamic window [v_min, v_max, omega_min, omega_max]
    """
    # Static constraints from robot specifications
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic constraints from current motion (acceleration limits)
    Vd = [x[3] - config.max_accel * config.dt,  # Min velocity reachable
          x[3] + config.max_accel * config.dt,  # Max velocity reachable
          x[4] - config.max_delta_yaw_rate * config.dt,  # Min yaw rate reachable
          x[4] + config.max_delta_yaw_rate * config.dt]  # Max yaw rate reachable

    # Intersection of static and dynamic constraints
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    Predict the trajectory for given constant velocity commands over prediction time.

    Simulates the robot motion forward in time using the kinematic model to
    generate a sequence of states that would result from applying constant
    linear velocity v and angular velocity y.

    Args:
        x_init: Initial state [x, y, yaw, v, omega]
        v: Linear velocity command
        y: Angular velocity command
        config: Configuration parameters

    Returns:
        trajectory: Array of predicted states over time
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goals, ob, current_goal_index):
    """
    Evaluate all possible trajectories in the dynamic window and select the optimal one.

    Samples velocity commands within the dynamic window, predicts trajectories,
    evaluates each using cost functions, and selects the trajectory with minimum cost.
    Includes a stuck detection mechanism to prevent the robot from getting trapped.

    Args:
        x: Current robot state
        dw: Dynamic window [v_min, v_max, omega_min, omega_max]
        config: Configuration parameters
        goal: Waypoint object representing the target
        ob: Obstacle positions

    Returns:
        best_u: Optimal control inputs [v, omega]
        best_trajectory: Predicted trajectory for optimal inputs
    """
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # Sample velocity commands within the dynamic window
    #print(np.arange(dw[0], dw[1], config.v_resolution))
    #print(np.arange(dw[2], dw[3], config.yaw_rate_resolution))
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)

            # Calculate costs for this trajectory
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goals, current_goal_index)
            speed_cost = config.speed_cost_gain * ((config.max_speed - trajectory[-1, 3])/config.max_speed) # Prefer higher speeds
            close_cost = config.close_cost_gain * dist_to_goal_cost(trajectory, goals, current_goal_index)
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(config, trajectory, ob)
            polygon_cost = calc_polygon_cost(trajectory, config.polygon)

            final_cost = to_goal_cost + speed_cost + ob_cost + close_cost + polygon_cost

            # Update best trajectory if this one has lower cost
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

                # Stuck detection: if robot is nearly stopped and facing goal,
                # force rotation to avoid getting stuck
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    print("Robot is stuck. Forcing rotation to escape.")
                    best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_trajectory


def calc_obstacle_cost(config, trajectory, ob):
    """
    Calculate obstacle avoidance cost for a trajectory.

    Returns infinity if collision detected, otherwise returns 1/min_distance
    to encourage keeping distance from obstacles.

    Args:
        trajectory: Predicted trajectory states
        ob: Obstacle positions array
        config: Configuration parameters

    Returns:
        Cost value (infinity for collision, 1/min_distance otherwise)
    """
    # Use Buoy.check_collision_with_boat for collision testing.
    # Pre-allocate arrays of buoy positions for distance computation
    buoy_x = np.array([b.x for b in ob])
    buoy_y = np.array([b.y for b in ob])

    # Temporary Boat used for collision checks at each trajectory state
    temp_boat = Boat(0, 0, 0)
    count = 0

    # Check collision for each point in the trajectory
    for pt in trajectory:
        temp_boat.x = float(pt[0])
        temp_boat.y = float(pt[1])
        temp_boat.theta = float(pt[2])

        for buoy in ob:
            if buoy.check_collision_with_boat(temp_boat):
                count += config.infinite_cost

    if count > 0:
        return count

    # No collision: compute minimum distance between trajectory points and buoys
    dx = trajectory[:, 0][:, None] - buoy_x[None, :]
    dy = trajectory[:, 1][:, None] - buoy_y[None, :]
    r = np.hypot(dx, dy)

    if r.size != 0:
        min_r = np.min(r)
        if min_r <= 0:
            return float("Inf")
        return min(1.0 / min_r, 1) # Normalisation dans [0, 1]
    return 0


def calc_polygon_cost(trajectory, polygon):
    """
    Calculate polygon boundary cost for a trajectory.

    Returns infinity if any point in the trajectory is outside the polygon,
    otherwise returns 0.

    Args:
        trajectory: Predicted trajectory states
        polygon: Polygon object representing the work area

    Returns:
        Cost value (infinity if any point is outside, 0 otherwise)
    """
    # Temporary Boat used for collision checks at each trajectory state
    temp_boat = Boat(0, 0, 0)
    count = 0
    
    # Check each point in the trajectory
    for pt in trajectory:
        temp_boat.x = float(pt[0])
        temp_boat.y = float(pt[1])
        temp_boat.theta = float(pt[2])
        if not polygon.is_boat_inside(temp_boat):
            count += config.infinite_cost

    return count


def calc_to_goal_cost(trajectory, goals, current_goal_index):
    """
    Calculate the cost for heading towards the goal based on angular difference.

    Computes the angle between the goal direction and the robot's final heading,
    normalized to [-pi, pi] range.

    Args:
        trajectory: Predicted trajectory
        goal: Waypoint object representing the target

    Returns:
        Angular cost (smaller when robot is facing towards goal)
    """

    goal = goals[current_goal_index]
    
    # Temporary Boat used for collision checks at each trajectory state
    temp_boat = Boat(0, 0, 0)
    
    # Check each point in the trajectory
    for pt in trajectory:
        temp_boat.x = float(pt[0])
        temp_boat.y = float(pt[1])
        temp_boat.theta = float(pt[2])
        if goal.check_collision_with_boat(temp_boat):
            return 0

    # Si on ne passe pas sur le goal, on fait avec l'angle

    # Last position in the trajectory
    pt = trajectory[-1]
    dx = pt[0]
    dy = pt[1]
    dtheta = pt[2]

    temp_boat.x = float(dx)
    temp_boat.y = float(dy)
    temp_boat.theta = float(dtheta)

    # Extract position from Waypoint
    gx, gy = goal.get_position(temp_boat)

    # Calculate desired heading towards goal
    dx = gx - trajectory[-1, 0]
    dy = gy - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)

    # Calculate difference between desired and actual heading
    cost_angle = error_angle - trajectory[-1, 2]

    # Normalize angle to [-pi, pi]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    cost = cost / math.pi # normalisation dans [0,1]

    return cost

def dist_to_goal_cost(trajectory, goals, current_goal_index):
    """
    Calculate the cost for being close to the goal.

    Args:
        trajectory: Predicted trajectory
        goal: Waypoint object representing the target

    Returns:
        Distance cost (smaller when robot is near goal)
    """

    goal = goals[current_goal_index]

    # Temporary Boat used for collision checks at each trajectory state
    temp_boat = Boat(0, 0, 0)
    
    # Check each point in the trajectory
    for index in range(len(trajectory)):
        pt = trajectory[index]
        temp_boat.x = float(pt[0])
        temp_boat.y = float(pt[1])
        temp_boat.theta = float(pt[2])
        if goal.check_collision_with_boat(temp_boat):
            if current_goal_index == len(goals) - 1:
                return 0 - current_goal_index
            else :
                return dist_to_goal_cost(trajectory[index:], goals, current_goal_index + 1)

    # Si on ne passe pas sur le goal, on fait avec la distance

    # Last position in the trajectory
    pt = trajectory[-1]
    dx = pt[0]
    dy = pt[1]
    dtheta = pt[2]

    temp_boat.x = float(dx)
    temp_boat.y = float(dy)
    temp_boat.theta = float(dtheta)

    # Extract position from Waypoint
    gx, gy = goal.get_position(temp_boat)

    # Euclidean Norm
    cost = np.linalg.norm(np.array([dx, dy])-np.array([gx, gy]))
    cost = min(cost, config.max_dist_for_cost)/config.max_dist_for_cost # normalisation dans [0,1]
    # Pas de coût supplémentaire au delà de config.max_dist_for_cost m
    return cost - current_goal_index # On veut aussi encourager à avancer dans les waypoints

def calculate_adaptive_speed(current_pos, goal_pos, base_speed, decel_distance=10.0, min_speed=0.3):
    """Réduit progressivement la vitesse en approchant le waypoint"""
    dist = math.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])
    
    if dist > decel_distance:
        return base_speed
    else:
        # Décélération linéaire de base_speed à min_speed
        speed_ratio = dist / decel_distance
        return min_speed + (base_speed - min_speed) * speed_ratio
