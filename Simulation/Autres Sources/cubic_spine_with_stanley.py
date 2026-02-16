# Import utilities
import pygame
import numpy as np
import sys
sys.path.insert(0, "./Pythonrobotics") # Pour les autres imports
from angle import angle_mod
import cubic_spline_planner as planner
import stanley_control_cropped as controller


# Autres Fichiers
from pygame_manager import * # initialise pygames et la cam

def main():
    """Plot an example of Stanley steering control on a cubic spline."""

    #################### Initialisation and unique settings ####################

    # Define target course waypoints
    ax = [0 , 10,   0,  -2.5,  -5,  10,  25, -10, 0.1] #in meters
    ay = [40,  0, -20, -25, -40, -60, -40, -10, 10] #in meters
        # Le premier point doit être proche de la position de départ du véhicule sinon le véhicule ne
        # va pas aller vers le bon point.

    # Generate smooth path using cubic splines
    cx, cy, cyaw, _, _ = planner.calc_spline_course(ax, ay, ds=0.1)

    # Target speed in m/s (30 km/h converted)
    target_speed = 10.0 / 3.6  # [m/s]

    # Initial vehicle state (starting slightly offset and rotated)
    state = controller.State(init_x, init_y, init_yaw, init_v)

    # Initialize tracking variables
    last_idx = len(cx) - 1  # Index of final point
    time = 0.0

    # Lists to store trajectory
    x, y, yaw, v, t = [state.x], [state.y], [state.yaw], [state.v], [time]

    # Find initial target point
    target_idx, _ = controller.calc_target_index(state, cx, cy)

    #################### Main loop ####################

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
        ai = controller.pid_control(target_speed, state.v)  # Acceleration (longitudinal)
        di, target_idx = controller.stanley_control(state, cx, cy, cyaw, target_idx)  # Steering (lateral)
        # Update vehicle state
        state.update(ai, di)

        # Update time and record state
        time += controller.dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # Render animation if enabled
        if show_animation:
            screen.fill(WHITE) # Reset window
            draw_basic_screen(screen) # Draw basic screen features such as the background, the border, the fixed buoy and the starting zone
            draw_reference_path(screen, cx, cy) # Draw reference path as red dots
            draw_waypoints(screen, ax, ay) # Draw waypoints as yellow dots
            draw_vehicle_trajectory(screen, x, y) # Draw vehicle trajectory as blue line
            draw_current_target_point(screen, cx, cy, target_idx) # Draw current target point as green circle
            draw_vehicle(screen, state) # Draw vehicle
            display_current_state(screen, state, time, font) # Display current state and time
            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS

    # Check if goal was reached
    if last_idx <= target_idx:
        goal_reached = True
        print("Goal reached")

    # Final display if goal reached
    if show_animation and goal_reached:
        screen.fill(WHITE) # Reset window
        draw_basic_screen(screen) # Draw basic screen features such as the background, the border, the fixed buoy and the starting zone
        draw_reference_path(screen, cx, cy) # Redraw path
        draw_vehicle_trajectory(screen, x, y) # Draw complete trajectory

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