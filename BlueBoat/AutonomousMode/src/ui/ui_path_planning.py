import math
import pygame
from boat_state import ThreadSafeBoatState
from src.path_planning_utils.simulation_parameters import parameters
from src.path_planning_utils.classes import Boat
from src.path_planning_utils.plot_simulation import *
import src.path_planning_utils.dynamic_window_approach_trimmed as planner

class UIPathPlanning:
    """
    Equivalent pygame de ton widget PySide6 Map.
    - Lit les variables via boat_state (tu ajusteras les clés comme tu veux).
    - API standardisée: update() + render(screen).
    """

    def __init__(self, boat_state: ThreadSafeBoatState, window_size = [parameters.window_width, parameters.window_height]):
        self.boat_state = boat_state
        parameters.window_width = window_size[0]
        parameters.window_height = window_size[1]
        parameters.window_scaling() # à appeler après avoir défini window_width et window_height

        # font (pygame doit être init avant ou le module font doit être init)
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont(None, 18)
    
    # ----------- standard API -----------
    def update(self):
        """
        Ici :
        - lire des paramètres dynamiques
        - faire des filtres / moyennes
        - gérer des animations
        """
        self.boat = self.boat_state["boatObject_pp"] # objet Boat de path_planning_utils/classes.py
        self.final_goal_reached = self.boat_state["bool_final_goal_reached_pp"] # simple bool pour savoir si le goal final a été atteint ou pas
        self.predicted_trajectory = self.boat_state["numpyArray_predicted_trajectory_pp"] # trajectoire prédite par le DWA pour les prochaines secondes
        self.waypoints = self.boat_state["waypointObjectList_pp"] # liste de waypoints à atteindre
        self.buoys = self.boat_state["buoyObjectList_pp"] # liste de bouées présentes sur la carte
        self.time = self.boat_state["time_pp"] # temps écoulé depuis le début de la simulation
        self.slam = self.boat_state["slamObject_pp"] # objet SLAM de path_planning_utils/classes.py, avec une méthode show_animation(screen) pour afficher les éléments du SLAM (carte, estimation de la pose, etc.)

        # Attention, màj seulement si terminé
        self.estimated_trajectory = self.boat_state["numpyArray_estimated_trajectory_pp"] # trajectoire estimée par le SLAM EKF
        self.true_trajectory = self.boat_state["numpyArray_true_trajectory_pp"] # trajectoire réelle du bateau (pour comparaison avec l'estimation)

        self.background_img, self.background_image_position = None, None


    def render(self, screen: pygame.Surface):

        if self.background_img is None or self.background_image_position is None:
            self.background_img, self.background_image_position = initialize_background()

        if not self.final_goal_reached:
            screen.fill(parameters.WHITE)
            draw_basic_screen(screen, self.background_img, self.background_image_position, self.boat.x, self.boat.y, self.boat.theta)

            for buoy in self.buoys:
                buoy.draw(screen, self.boat.x, self.boat.y, self.boat.theta)

            for waypoint in self.waypoints:
                waypoint.draw(screen, self.font, self.boat.x, self.boat.y, self.boat.theta)

            temp_x = self.predicted_trajectory[:, 0]
            temp_y = self.predicted_trajectory[:, 1]
            draw_vehicle_trajectory(screen, temp_x, temp_y, self.boat.x, self.boat.y, self.boat.theta)

            # Draw boat
            self.boat.show(screen)
            self.boat.display_state(screen, self.time, self.font)

            # Draw heading arrow
            self.boat.draw_arrow(screen)

            self.slam.show_animation(screen)

            planner.config.polygon.draw(screen, self.boat.x, self.boat.y, self.boat.theta)

        # Display final trajectory when goal reached
        if self.final_goal_reached:
            screen.fill(parameters.WHITE)
            draw_basic_screen(screen, self.background_img, self.background_image_position, self.boat.x, self.boat.y, self.boat.theta)

            for buoy in self.buoys:
                buoy.draw(screen, self.boat.x, self.boat.y, self.boat.theta)

            for waypoint in self.waypoints:
                waypoint.draw(screen, self.font, self.boat.x, self.boat.y, self.boat.theta)
            
            temp_x = self.estimated_trajectory[:, 0]
            temp_y = self.estimated_trajectory[:, 1]
            draw_reference_path(screen, temp_x, temp_y, self.boat.x, self.boat.y, self.boat.theta)

            temp_x = self.true_trajectory[:, 0]
            temp_y = self.true_trajectory[:, 1]
            draw_reference_path(screen, temp_x, temp_y, self.boat.x, self.boat.y, self.boat.theta, color=parameters.BLUE)

            planner.config.polygon.draw(screen, self.boat.x, self.boat.y, self.boat.theta)


        
