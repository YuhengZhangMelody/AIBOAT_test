# Import utilities
import pygame
import numpy as np
import math
import sys
sys.path.insert(0, "./Pythonrobotics") # Pour les autres imports
import dynamic_window_approach_trimmed as planner

from simulation_parameters import parameters

# Autres Fichiers
from pygame_manager import * # initialise pygames et la cam
from classes import * # Définit les bouées, les waypoints et le bateau
import ekf_slam

def main():
    """
    Main simulation function demonstrating Dynamic Window Approach navigation.


    Sets up the simulation environment, runs the control loop with Pygame visualization,
    and displays the final trajectory when the goal is reached.


    Args:
        gx, gy: Goal position coordinates
        robot_type: Shape of the robot (circle or rectangle)
    """


    #################### Initialisation and unique settings ####################


    print(__file__ + " start!!")

    ####### Elements de la simulation #######

    # Load Waypoints
    waypoints = []
    number = 1
    for waypoint in parameters.waypoint_positions:
        if isinstance(waypoint, list):
            # If waypoint is a list of tuples, treat each tuple as a point
            point1 = waypoint[0]
            point2 = waypoint[1]
            waypoints.append(Waypoint(point1[0], point1[1], waypoint_type='goal', number=number, waypoint_shape='segment', x2=point2[0], y2=point2[1]))
        else:
            # If waypoint is a tuple (x, y)
            waypoint_pos_x, waypoint_pos_y = waypoint
            waypoints.append(Waypoint(waypoint_pos_x, waypoint_pos_y, waypoint_type='goal', number=number))
        number += 1
    waypoints_nb = 0
    current_goal = waypoints[waypoints_nb] # First Goal

    # Load Buoys
    buoys = []
    for buoy in parameters.buoy_positions :
        buoys.append(Buoy(buoy[0], buoy[1])) # Buoys

    # Modèle de bateau pour la simulation
    boat = Boat(parameters.init_x, parameters.init_y, parameters.init_yaw, alpha=150)
    boat.v = parameters.init_v
    boat.omega = parameters.init_yaw

    ####### DWA #######

    # Modèle de robot pour dwa. [x, y, yaw, v, omega] # A SUPPRIMER, utiliser plutot la classe boat
    x = np.array([parameters.init_x, parameters.init_y, parameters.init_yaw, parameters.init_v, parameters.init_yaw])

    # No initial command
    u = [0, 0]

    ####### SLAM #######
    
    # Ajout des bouées dans le SLAM
    buoys_slam = []
    for buoy in parameters.buoy_positions :
        buoys_slam.append([buoy[0], buoy[1]]) # Buoys

    # Lancement du SLAM
    slam = ekf_slam.Slam(np.array(buoys_slam).reshape(2, len(parameters.buoy_positions)), np.array([parameters.init_x, parameters.init_y, parameters.init_yaw]).reshape(3, 1), boat)

    ####### AUTRES #######

    # Historique des positions pour l'affichage final
    estimated_trajectory = np.array([boat.x, boat.y])
    true_trajectory = np.array([boat.x, boat.y])
    time = 0.0

    #################### Main loop ####################


    running = True
    final_goal_reached = False
    # Main simulation loop
    while running and not final_goal_reached:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False



        ### --------- CALCUL --------- ####

        ####### SLAM #######

        # Get estimation of pose for the boat and the buoys
        xEst, PEst = slam.get_estimate_full_motion(np.array([u]).T)
        buoysEst = np.array(xEst[3:, :])
        if buoysEst.size==0:
            # No updates from SLAM; keep original buoys positions
            buoysEst = np.array([[], []]).T
        else : 
            buoysEst = buoysEst.reshape(-1, 2)
        pose = xEst[:, 0]

        x = [pose[0], pose[1], pose[2], x[3], x[4]] # Update state with SLAM estimation

        ####### DWA #######
                
        # Ajuste la vitesse max selon la distance au waypoint
        if waypoints_nb == len(waypoints) - 1:
            # Si on est sur le dernier waypoint, on peut se permettre de ralentir plus tôt pour une approche plus douce
            goal_position = current_goal.get_position()
            adaptive_max_speed = planner.calculate_adaptive_speed(
                [x[0], x[1]], 
                goal_position, 
                parameters.vehicle_max_speed,        # Vitesse normale d'origine
                decel_distance=10.0,       # Commence à ralentir à 10m
                min_speed=parameters.vehicle_max_speed * 0.7          # Vitesse minimum à atteindre
            )
            planner.config.max_speed = adaptive_max_speed      

        # Compute optimal control inputs using DWA
        # Ajout des bouées estimées dans le DWA
        buoysEst_dwa = []
        for buoyEst in buoysEst :
            buoysEst_dwa.append(Buoy(buoyEst[0], buoyEst[1])) # Buoys Class
        u, predicted_trajectory = planner.dwa_control(x, planner.config, waypoints, buoysEst_dwa, waypoints_nb)

        # Update robot state with optimal inputs
        x = planner.motion(x, u, planner.config.dt)

        ####### SIMULATION #######

        # Update persistent boat object pose/velocities
        try:
            boat.x = x[0]
            boat.y = x[1]
            boat.theta = x[2]
            boat.v = x[3]
            boat.omega = x[4]
            # maintain trail
            boat.trail.append((int(boat.x), int(boat.y)))
            if len(boat.trail) > boat.max_trail_length:
                boat.trail.pop(0)
        except NameError:
            # boat not initialized (should not happen)
            pass

        estimated_trajectory = np.vstack((estimated_trajectory, [boat.x, boat.y]))  # Store state history
        true_trajectory = np.vstack((true_trajectory, [boat.x_true, boat.y_true]))  # Store state history

        ### --------- AFFICHAGE --------- ####

        # Render visualization if enabled
        if parameters.show_animation:
            screen.fill(parameters.WHITE)
            draw_basic_screen(screen, boat.x, boat.y, boat.theta)

            for buoy in buoys:
                buoy.draw(screen, boat.x, boat.y, boat.theta)

            for waypoint in waypoints:
                waypoint.draw(screen, font, boat.x, boat.y, boat.theta)

            temp_x = predicted_trajectory[:, 0]
            temp_y = predicted_trajectory[:, 1]
            draw_vehicle_trajectory(screen, temp_x, temp_y, boat.x, boat.y, boat.theta)

            # Draw boat
            boat.show(screen)
            boat.display_state(screen, time, font)

            # Draw heading arrow
            boat.draw_arrow(screen)

            slam.show_animation(screen)

            planner.config.polygon.draw(screen, boat.x, boat.y, boat.theta)

            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS

        # Check if goal is reached using Waypoint collision check with boat
        if current_goal.check_collision_with_boat(boat):
            # mark current goal as reached for visualization/state
            current_goal.reached = True

            waypoints_nb += 1
            print("Goal!!")
            if waypoints_nb >= len(waypoints):
                final_goal_reached = True
            else:
                current_goal = waypoints[waypoints_nb]
                planner.config.max_speed = parameters.vehicle_max_speed
        
        # Update time
        time += planner.config.dt

    print("Done")


    # Display final trajectory when goal reached
    if parameters.show_animation and final_goal_reached:
        screen.fill(parameters.WHITE)
        draw_basic_screen(screen, boat.x, boat.y, boat.theta)

        for buoy in buoys:
            buoy.draw(screen, boat.x, boat.y, boat.theta)

        for waypoint in waypoints:
            waypoint.draw(screen, font, boat.x, boat.y, boat.theta)
        
        temp_x = estimated_trajectory[:, 0]
        temp_y = estimated_trajectory[:, 1]
        draw_reference_path(screen, temp_x, temp_y, boat.x, boat.y, boat.theta)

        temp_x = true_trajectory[:, 0]
        temp_y = true_trajectory[:, 1]
        draw_reference_path(screen, temp_x, temp_y, boat.x, boat.y, boat.theta, color=parameters.BLUE)

        planner.config.polygon.draw(screen, boat.x, boat.y, boat.theta)

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


if __name__ == '__main__':
    import argparse

    # Initialize the parser
    parser = argparse.ArgumentParser(description="A script with flags")
    parser.add_argument("-rt", "--realtime", action="store_true", help="Enable realtime simulation")
    args = parser.parse_args()
    parameters.realtime = args.realtime
    main()
