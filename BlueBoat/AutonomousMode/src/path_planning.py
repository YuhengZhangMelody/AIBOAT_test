# Import utilities
from threading import Thread
from pathlib import Path
from boat_state import ThreadSafeBoatState
import sys

#Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import math

import src.path_planning_utils.dynamic_window_approach_trimmed as planner

from src.path_planning_utils.simulation_parameters import parameters

# Autres Fichiers
from src.path_planning_utils.classes import * # Définit les bouées, les waypoints et le bateau
import src.path_planning_utils.ekf_slam as ekf_slam # Définit la classe de SLAM EKF

class PathPlanning(Thread):

    LOOP_SPEED_HZ = 10
    parameters.realtime = True # Permet de faire tourner la simulation en temps réel (sleep pour respecter la boucle à 10Hz). A mettre à False pour faire tourner la simulation plus rapidement que le temps réel (pas de sleep, juste une boucle while qui tourne tant que la simulation n'est pas terminée)

    def __init__(self, boat_state: ThreadSafeBoatState):
        """
        Main simulation function demonstrating Dynamic Window Approach navigation.


        Sets up the simulation environment, runs the control loop with Pygame visualization,
        and displays the final trajectory when the goal is reached.


        Args:
            gx, gy: Goal position coordinates
            robot_type: Shape of the robot (circle or rectangle)
        """

        self.boat_state = boat_state # Pour le partage de données

        #################### Initialisation and unique settings ####################

        ####### Elements de la simulation #######

        # Load Waypoints
        self.waypoints = []
        number = 1
        for waypoint in parameters.waypoint_positions:
            if isinstance(waypoint, list):
                # If waypoint is a list of tuples, treat each tuple as a point
                point1 = waypoint[0]
                point2 = waypoint[1]
                self.waypoints.append(Waypoint(point1[0], point1[1], waypoint_type='goal', number=number, waypoint_shape='segment', x2=point2[0], y2=point2[1]))
            else:
                # If waypoint is a tuple (x, y)
                waypoint_pos_x, waypoint_pos_y = waypoint
                self.waypoints.append(Waypoint(waypoint_pos_x, waypoint_pos_y, waypoint_type='goal', number=number))
            number += 1
        self.waypoints_nb = 0
        self.current_goal = self.waypoints[self.waypoints_nb] # First Goal

        # Load Buoys
        self.buoys = []
        for buoy in parameters.buoy_positions :
            self.buoys.append(Buoy(buoy[0], buoy[1])) # Buoys

        # Modèle de bateau pour la simulation
        self.boat = Boat(parameters.init_x, parameters.init_y, parameters.init_yaw, alpha=150)
        self.boat.v = parameters.init_v
        self.boat.omega = parameters.init_yaw

        ####### DWA #######

        # Modèle de robot pour dwa. [x, y, yaw, v, omega] # A SUPPRIMER, utiliser plutot la classe boat
        self.x = np.array([parameters.init_x, parameters.init_y, parameters.init_yaw, parameters.init_v, parameters.init_yaw])

        # No initial command
        self.u = [0, 0]

        ####### SLAM #######
        
        # Ajout des bouées dans le SLAM
        buoys_slam = []
        for buoy in parameters.buoy_positions :
            buoys_slam.append([buoy[0], buoy[1]]) # Buoys

        # Lancement du SLAM
        self.slam = ekf_slam.Slam(np.array(buoys_slam).reshape(2, len(parameters.buoy_positions)),
                                   np.array([parameters.init_x, parameters.init_y, parameters.init_yaw]).reshape(3, 1),
                                     self.boat)

        ####### AUTRES #######

        # Historique des positions pour l'affichage final
        self.estimated_trajectory = np.array([self.boat.x, self.boat.y])
        self.true_trajectory = np.array([self.boat.x, self.boat.y])
        self.time = 0.0

        self.running = True # Rajouter un signal de départ
        self.final_goal_reached = False

        #################### Diffusion ####################

        self.boat_state["slamObject_pp"] = self.slam # objet SLAM de path_planning_utils/classes.py, avec une méthode show_animation(screen) pour afficher les éléments du SLAM (carte, estimation de la pose, etc.)
        self.boat_state["waypointObjectList_pp"] = self.waypoints
        self.boat_state["buoyObjectList_pp"] = self.buoys
        self.boat_state["boatObject_path_planning"] = self.boat

        #################### Récupération infos initialisation ####################

        # Initialisation GPS

        self.init_lat = self.boat_state["latitude"]
        self.init_lon = self.boat_state["longitude"]
        self.init_north = self.boat_state["heading"] # Pas sûr de ça

    def step(self):
        # Main simulation loop
        if self.running and not self.final_goal_reached:

             ### --------- Récupération --------- ####

            if parameters.realtime:
                # Observations de caméra et GNSS

                # Caméra
                y = np.zeros((0, 3))
                for measurement in self.boat_state["detected_objects"]: # format (pt_x, pt_y, pt_z, dist)
                    pt_x, pt_y, _, dist = measurement
                    angle_n = math.atan2(pt_y, pt_x) - self.boat.theta
                    yi = np.array([dist, angle_n, 0])
                    y = np.vstack((y, yi))

                # Gnss
                # Convertit les coordonnées GPS en coordonnées locales (x, y) par rapport à la position initiale du bateau
                dx, dy = self.slam.gps_to_local(self.init_lat, self.init_lon, self.init_north, self.boat_state["latitude"], self.boat_state["longitude"])
                xGnss = np.array([[dx+10], [dy]]) + np.array([[parameters.init_x], [parameters.init_y]])
                # xGnss = np.array([[0.0], [0.0]])  # Par défaut, à remplacer par les vraies observations GNSS [x, y] format (2x1)
            else:
                # En mode non temps réel, on suppose que les observations sont générées par le SLAM lui même à partir de l'état vrai, donc on n'a pas besoin de les récupérer depuis le boat_state, on les génère directement dans la fonction observation_model du SLAM.
                y = None
                xGnss = None


            ### --------- CALCUL --------- ####

            ####### SLAM #######

            # Get estimation of pose for the boat and the buoys
            xEst, PEst = self.slam.get_estimate_full_motion(np.array([self.u]).T, y, xGnss)
            buoysEst = np.array(xEst[3:, :])
            if buoysEst.size==0:
                # No updates from SLAM; keep original buoys positions
                buoysEst = np.array([[], []]).T
            else : 
                buoysEst = buoysEst.reshape(-1, 2)
            pose = xEst[:, 0]

            x = [pose[0], pose[1], pose[2], self.x[3], self.x[4]] # Update state with SLAM estimation

            ####### DWA #######
                    
            # Ajuste la vitesse max selon la distance au waypoint
            if self.waypoints_nb == len(self.waypoints) - 1:
                # Si on est sur le dernier waypoint, on peut se permettre de ralentir plus tôt pour une approche plus douce
                goal_position = self.current_goal.get_position()
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
            u, predicted_trajectory = planner.dwa_control(x, planner.config, self.waypoints, buoysEst_dwa, self.waypoints_nb)
            u = [10, u[1]] # format (acceleration, steering_angle)

            # Update robot state with optimal inputs
            x = planner.motion(x, u, planner.config.dt)
            
            ####### SIMULATION #######

            # Update persistent boat object pose/velocities
            try:
                self.boat.x = x[0]
                self.boat.y = x[1]
                self.boat.theta = x[2]
                self.boat.v = x[3]
                self.boat.omega = x[4]
                # maintain trail
                self.boat.trail.append((int(self.boat.x), int(self.boat.y)))
                if len(self.boat.trail) > self.boat.max_trail_length:
                    self.boat.trail.pop(0)
            except NameError:
                # boat not initialized (should not happen)
                pass

            self.estimated_trajectory = np.vstack((self.estimated_trajectory, [self.boat.x, self.boat.y]))  # Store state history
            self.true_trajectory = np.vstack((self.true_trajectory, [self.boat.x_true, self.boat.y_true]))  # Store state history

            # Check if goal is reached using Waypoint collision check with boat
            if self.current_goal.check_collision_with_boat(self.boat):
                # mark current goal as reached for visualization/state
                self.current_goal.reached = True

                self.waypoints_nb += 1
                print("Goal!!")
                if self.waypoints_nb >= len(self.waypoints):
                    self.final_goal_reached = True
                else:
                    self.current_goal = self.waypoints[self.waypoints_nb]
                    planner.config.max_speed = parameters.vehicle_max_speed
            
            # Update time
            self.time += planner.config.dt

            ### --------- DIFFUSION --------- ####
            # Transmission des infos d'affichages uniquement
            self.boat_state["boatObject_path_planning"] = self.boat
            self.boat_state["bool_final_goal_reached_pp"] = self.final_goal_reached
            self.boat_state["numpyArray_predicted_trajectory_pp"] = predicted_trajectory
            self.boat_state["time_pp"] = self.time

        elif self.running and self.final_goal_reached:
            self.boat_state["numpyArray_estimated_trajectory_pp"] = self.estimated_trajectory # trajectoire estimée par le SLAM EKF
            self.boat_state["numpyArray_true_trajectory_pp"] = self.true_trajectory