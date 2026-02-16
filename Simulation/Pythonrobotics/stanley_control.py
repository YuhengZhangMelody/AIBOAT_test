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
import cubic_spline_planner

# Stanley controller parameters
k = 0.5  # control gain for cross-track error
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time step
L = 2.9  # [m] Wheel base of vehicle (distance between front and rear axles)
max_steer = np.radians(30.0)  # [rad] max steering angle

# Simulation settings
show_animation = True

# Pygame visualization settings
scale = 5  # pixels per meter
window_width = 600
window_height = 600


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


def draw_vehicle(screen, state):
    """
    Draw the vehicle as a blue rectangle on the Pygame screen.

    The vehicle is represented as a rectangle rotated according to its yaw angle.
    """
    # Vehicle dimensions in meters
    vehicle_length = 4.5  # meters
    vehicle_width = 2.0   # meters
    # Convert to pixels
    half_length = vehicle_length / 2 * scale
    half_width = vehicle_width / 2 * scale
    # Define rectangle corners relative to center
    points = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width)
    ]
    # Rotation matrix for yaw (note: sin_yaw negated for correct orientation in Pygame)
    cos_yaw = np.cos(state.yaw)
    sin_yaw = -np.sin(state.yaw)
    rotated_points = []
    for px, py in points:
        # Rotate and translate to vehicle position
        rx = px * cos_yaw - py * sin_yaw + state.x * scale
        ry = px * sin_yaw + py * cos_yaw + (window_height - state.y * scale)
        rotated_points.append((rx, ry))
    pygame.draw.polygon(screen, (0, 0, 255), rotated_points)  # Blue vehicle


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


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    # Define target course waypoints
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    # Generate smooth path using cubic splines
    cx, cy, cyaw, _, _ = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    # Target speed in m/s (30 km/h converted)
    target_speed = 30.0 / 3.6  # [m/s]

    # Simulation parameters
    max_simulation_time = 100.0

    # Initial vehicle state (starting slightly offset and rotated)
    state = State(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0)

    # Initialize tracking variables
    last_idx = len(cx) - 1  # Index of final point
    time = 0.0
    # Lists to store trajectory
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    # Find initial target point
    target_idx, _ = calc_target_index(state, cx, cy)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Stanley Control Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Define colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)

    # Main simulation loop
    running = True
    goal_reached = False
    while running and max_simulation_time >= time and last_idx > target_idx:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Compute control inputs
        ai = pid_control(target_speed, state.v)  # Acceleration (longitudinal)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)  # Steering (lateral)
        # Update vehicle state
        state.update(ai, di)

        # Update time and record state
        time += dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # Render animation if enabled
        if show_animation:
            screen.fill(WHITE)

            # Draw reference path as red dots
            for i in range(len(cx)):
                cx_p = int(cx[i] * scale)
                cy_p = int(window_height - cy[i] * scale)
                pygame.draw.circle(screen, RED, (cx_p, cy_p), 2)

            # Draw vehicle trajectory as blue line
            if len(x) > 1:
                for i in range(1, len(x)):
                    tx1 = int(x[i-1] * scale)
                    ty1 = int(window_height - y[i-1] * scale)
                    tx2 = int(x[i] * scale)
                    ty2 = int(window_height - y[i] * scale)
                    pygame.draw.line(screen, BLUE, (tx1, ty1), (tx2, ty2), 2)

            # Draw current target point as green circle
            target_x = int(cx[target_idx] * scale)
            target_y = int(window_height - cy[target_idx] * scale)
            pygame.draw.circle(screen, GREEN, (target_x, target_y), 5)

            # Draw vehicle
            draw_vehicle(screen, state)

            # Display current speed
            speed_text = font.render(f"Speed: {state.v * 3.6:.2f} km/h", True, BLACK)
            screen.blit(speed_text, (10, 10))

            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS

    # Check if goal was reached
    if last_idx <= target_idx:
        goal_reached = True
        print("Goal reached")

    # Final display if goal reached
    if show_animation and goal_reached:
        screen.fill(WHITE)
        # Redraw path
        for i in range(len(cx)):
            cx_p = int(cx[i] * scale)
            cy_p = int(window_height - cy[i] * scale)
            pygame.draw.circle(screen, RED, (cx_p, cy_p), 2)
        # Draw complete trajectory
        for i in range(1, len(x)):
            tx1 = int(x[i-1] * scale)
            ty1 = int(window_height - y[i-1] * scale)
            tx2 = int(x[i] * scale)
            ty2 = int(window_height - y[i] * scale)
            pygame.draw.line(screen, BLUE, (tx1, ty1), (tx2, ty2), 2)
        pygame.display.flip()

        # Wait for user to close window
        waiting = True
        while waiting and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False

    pygame.quit()


if __name__ == '__main__':
    main()
