"""

Path tracking simulation with Stanley steering control and PID speed control.

This module simulates a vehicle following a path using the Stanley steering controller,
which combines heading error and cross-track error for lateral control, and PID for longitudinal control.

author: Atsushi Sakai (@Atsushi_twi)

Reference:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import numpy as np
import pygame
import sys
import pathlib

# Import utilities
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from angle import angle_mod

# Stanley controller parameters
k = 0.5  # control gain for cross-track error
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time step

# Vehicle parameters
L = 2.9  # [m] Wheel base of vehicle (distance between front and rear axles)
max_steer = np.radians(30.0)  # [rad] max steering angle

class State:
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle (heading)
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super().__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle using the bicycle model.

        The bicycle model approximates the vehicle as having front and rear wheels
        on a single axis, with steering only at the front.

        :param acceleration: (float) Longitudinal acceleration
        :param delta: (float) Steering angle
        """
        # Limit steering angle to maximum
        delta = np.clip(delta, -max_steer, max_steer)

        # Update position using current velocity and heading
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        # Update heading using bicycle model kinematics
        self.yaw += self.v / L * np.tan(delta) * dt
        # Normalize yaw to [-pi, pi]
        self.yaw = normalize_angle(self.yaw)
        # Update speed
        self.v += acceleration * dt

def pid_control(target, current):
    """
    Proportional control for the speed (longitudinal control).

    This is a simple P-controller that computes acceleration based on speed error.

    :param target: (float) Target speed
    :param current: (float) Current speed
    :return: (float) Acceleration command
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control for lateral control.

    The Stanley controller combines two terms:
    1. Heading error: difference between desired and current yaw
    2. Cross-track error: lateral distance to path, projected onto front axle

    :param state: (State object) Current vehicle state
    :param cx: ([float]) x-coordinates of the reference path
    :param cy: ([float]) y-coordinates of the reference path
    :param cyaw: ([float]) yaw angles of the reference path
    :param last_target_idx: (int) Previous target index to ensure monotonic progress
    :return: (float, int) Steering angle delta and current target index
    """
    # Find the target point on the path (closest to front axle)
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    # Ensure we don't go backwards on the path
    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # Heading error: difference between path yaw and vehicle yaw
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # Cross-track error correction: atan(k * lateral_error / speed)
    # This term approaches 90 degrees when error is large, and 0 when error is small
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Total steering: sum of heading and cross-track corrections
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return angle_mod(angle)


def calc_target_index(state, cx, cy):
    """
    Compute the index in the trajectory list of the target point.

    The target is the point on the path closest to the vehicle's front axle.
    Also computes the cross-track error projected onto the front axle vector.

    :param state: (State object) Current vehicle state
    :param cx: [float] x-coordinates of path
    :param cy: [float] y-coordinates of path
    :return: (int, float) Target index and front axle error
    """
    # Calculate front axle position (L meters ahead of rear axle)
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Compute distances from front axle to all path points
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    # Find the closest point
    target_idx = np.argmin(d)

    # Project the error vector onto the front axle lateral direction
    # Front axle vector is perpendicular to heading
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle
