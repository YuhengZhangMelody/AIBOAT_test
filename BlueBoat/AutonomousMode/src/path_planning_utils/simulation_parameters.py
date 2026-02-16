import os # Pour les chemins de fichiers
import math
import numpy as np

"""
This file contains common basic parameters for most displays
"""

class Parameters:
    """
    Simulation parameter class containing all configuration parameters for DWA.

    This class holds robot physical constraints, cost function weights, obstacle
    positions, and visualization settings used throughout the simulation.
    """
    
    ##################### Pygame Colors (Class attributes) ####################
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 100, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    DARK_RED = (139, 0, 0)
    DARK_GREEN = (0, 200, 0)
    PURPLE = (128, 0, 128)
    CYAN = (0, 255, 255)

    def __init__(self):
        """
        CECI EST LA CONFIG PAR DEFAUT
        POUR CREER SA PROPRE CONFIG, IL FAUT CREER SA FONCTION ET ECRASER LES PARAMETRES EXISTANTS
        N'OUBLIEZ PAS D'APPELLER self.window_scaling() A LA FIN DE LA FONCTION SI VOUS AVEZ CHANGé L'AFFICHAGE
        UTILISEZ ENSUITE LA FONCTION A LA FIN DE CE FICHIER
        """

        #################### Simulation settings ####################
        self.show_animation = True
        self.realtime = False # Change le comportement des observations
        self.max_simulation_time = 100.0
        self.simulation_height = (-80, 50)  # in meters (= limite réelle minimale)
        self.simulation_width = (-80, 80) # in meters  (= limite réelle minimale)

        # Inital state
        self.init_x = 0.0 # in meters
        self.init_y = 40.0 # in meters
        self.init_yaw = np.radians(-90.0) # in radians
        self.init_v = 0.0 # in m/s
        self.init_omega = 0.0 # in rad/s (=\dot{yaw})

        #################### DWA settings ####################
        self.sampling_v_resolution = 0.01  # [m/s] Resolution for velocity sampling
        self.sampling_yaw_rate_resolution = 1 * math.pi / 180.0  # [rad/s] Resolution for yaw rate sampling
        self.time_step = 0.1  # [s] Time step for motion prediction
        self.time_horizon = 0.5 # [s] Prediction horizon for trajectory evaluation

        # Cost function weights
        self.to_goal_cost_gain = 2.0  # Weight for goal-oriented cost
        self.speed_cost_gain = 0.001  # Weight for preferring higher speeds
        self.obstacle_cost_gain = 0.01  # Weight for obstacle avoidance
        self.close_cost_gain = 2.0 # Weight for goal closeness
        self.max_dist_for_cost = 10 # [m] distance maximale après laquelle le cout est le même

        #################### Object settings ####################
        # Buoy parameters
        self.buoy_positions = [(0, 0), (10.25, -54.75)] # in meters
        self.buoy_radius = 1.5  # in meters
        # Vehicle parameters
        self.vehicle_length = 4.5  # meters
        self.vehicle_width = 2.0   # meters
        self.vehicle_max_speed = 10.0  # [m/s] Maximum linear velocity
        self.vehicle_min_speed = -1  # [m/s] Minimum linear velocity (allows backward motion)
        self.vehicle_max_yaw_rate = 45.0 * math.pi / 180.0  # [rad/s] Maximum angular velocity
        self.vehicle_max_accel = 1  # [m/ss] Maximum linear acceleration
        self.vehicle_max_delta_yaw_rate = 45.0 * math.pi / 180.0  # [rad/ss] Maximum angular acceleration
        # Waypoint parameters
        self.waypoint_positions = [(10.0, 0.0), (0.0, -20.0), (-2.5, -25.0), (-5.0, -50.0), (7.0, -70.0), (20.0, -70.0), (25.0, -40.0), (-10.0, -10.0), (self.init_x, self.init_y)]
            # in meters dans l'ordre des waypoints
            # Des tuples (x,y) pour les points, ou de listes de tuples [(x1,y1),(x2,y2)] pour les segments
        self.waypoint_radius = 2  # in meters
        # Starting zone parameters
        self.start_zone = ((-5, 45), (5, 35)) # x_top_left_x, y_top_left_y, x_bottom_right_x, y_bottom_right_y  (in meters)
        # Polygon (work area) parameters
        self.polygon_points = [(-55, 45), (40, 45), (75, -75), (0, -75)]  # in meters (rectangle containing the work area)

        #################### Pygame visualization settings ##################### 
        self.window_width = 1200 # pixels (= taille de la fenêtre)
        self.window_height = 800 # pixels (= taille de la fenêtre)
        self.border_offset_top = 0  # pixels (= margin around the simulation area)
        self.border_offset_left = 210  # pixels
        self.border_offset_right = 0  # pixels
        self.border_offset_bottom = 0  # pixels
        self.background_image_path = os.path.join("Images", "Port de monaco 2.png")
        self.background_image_zero_position_local = (575, 417)  # pixels
        # Il faut ajuster cette position pour que l'origine (0,0) de la simulation corresponde au bon point sur l'image
        # L'origine (0,0) de la simulation est la bouée la plus proche du coin supérieur gauche de l'image
        self.background_image_scale_local = 100.0/20.0
        
        #################### View settings ####################
        self.view_type = 'static'  # 'static' or 'boat' (boat-centered view)
        self.zoom = 1  # Zoom factor for boat view (pixels per meter)
        
        self.window_scaling()

    def window_scaling(self):
        #----------------- Scaling the window  -----------------#
        if self.view_type == 'boat':
            # Boat-centric view: boat stays in center, use zoom for scaling
            self.scale = self.zoom  # pixels per meter
            self.center_offset = (self.window_width / 2, self.window_height / 2)  # Center of window
            # Background should still be scaled relative to the local image scale
            # so we can render the world background in boat view. Precompute factor.
            self.factor = self.scale / self.background_image_scale_local
            # background_image_zero_position is kept in local-image pixels; top-left
            # on-screen position depends on boat pose, so we keep the local reference.
            self.background_image_zero_position = (self.center_offset[0] - self.background_image_zero_position_local[0] * self.factor,
                                                  self.center_offset[1] - self.background_image_zero_position_local[1] * self.factor)
        else:
            # Static view: traditional centered view
            #compute scale factor to convert meters to pixels
            scale_height = (self.window_height - self.border_offset_top - self.border_offset_bottom) // (self.simulation_height[1] - self.simulation_height[0])  # pixels per meter
            scale_width = (self.window_width - self.border_offset_left - self.border_offset_right)  // (self.simulation_width[1] - self.simulation_width[0])  # pixels per meter
            self.scale = min(scale_height, scale_width) # pixels per meter
            # Scale factor for background image to match simulation scale
            self.factor = self.scale / self.background_image_scale_local
            # Center the simulation area in the window (local variable)
            center = ((self.simulation_width[1] + self.simulation_width[0]) // 2, (self.simulation_height[1] + self.simulation_height[0]) // 2) # in meters
            self.center_offset = (self.window_width // 2 - center[0]*self.scale + self.border_offset_left/self.scale, - self.window_height // 2 + center[1]*self.scale + self.border_offset_bottom)  # in pixels
            # ^--- Avec décalage du centre pour laisser les bordures
            # Position du fond
            self.background_image_zero_position = (self.center_offset[0] - self.background_image_zero_position_local[0] * self.factor, self.window_height + self.center_offset[1] - self.background_image_zero_position_local[1] * self.factor)  # pixels

    def get_screen_position(self, x, y, boat_x=None, boat_y=None, boat_theta=None):
        """
        Convert world coordinates (x, y) to screen coordinates.
        
        Args:
            x: world x coordinate (meters)
            y: world y coordinate (meters)
            boat_x: boat x position (required for 'boat' view)
            boat_y: boat y position (required for 'boat' view)
            boat_theta: boat heading angle (required for 'boat' view with world rotation)
        
        Returns:
            Tuple (screen_x, screen_y) in pixels
        """
        if self.view_type == 'boat':
            # Boat-centric view: boat is always at center, world rotates around it
            if boat_x is None or boat_y is None:
                raise ValueError("boat_x and boat_y required for 'boat' view mode")
            
            # Calculate position relative to boat
            rel_x = (x - boat_x) * self.scale
            rel_y = (boat_y - y) * self.scale  # Flip y-axis for Pygame
            
            # Apply rotation inverse to boat heading if provided (world rotates opposite to boat)
            if boat_theta is not None:
                # Rotate by boat_theta
                cos_theta = math.cos(boat_theta - np.radians(90))
                sin_theta = math.sin(boat_theta - np.radians(90))
                rotated_x = rel_x * cos_theta - rel_y * sin_theta
                rotated_y = rel_x * sin_theta + rel_y * cos_theta
            else:
                rotated_x = rel_x
                rotated_y = rel_y
            
            # Convert to screen coordinates with boat at center
            screen_x = self.center_offset[0] + rotated_x
            screen_y = self.center_offset[1] + rotated_y
        else:
            # Static view: standard transformation
            screen_x = x * self.scale + self.center_offset[0]
            screen_y = self.window_height - y * self.scale + self.center_offset[1]
        
        return (screen_x, screen_y)

    def test_affichage(self):
        #################### Simulation settings ####################
        self.simulation_height = (-20, 20)  # in meters (= limite réelle minimale)
        self.simulation_width = (-20, 20) # in meters  (= limite réelle minimale)

        # Inital state
        self.init_x = 5.0 # in meters
        self.init_y = 5.0 # in meters
        self.init_yaw = np.radians(-100.0) # in radians

        #################### Object settings ####################
        # Buoy parameters
        self.buoy_positions = [(0, 0), (1, 1)] # in meters
        self.buoy_radius = 1.5  # in meters
        # Vehicle parameters
        self.vehicle_length = 4.5  # meters
        self.vehicle_width = 2.0   # meters
        # Waypoint parameters
        self.waypoint_positions = [(-5, -5),[(12, 0), (12, 15)]]
            # in meters dans l'ordre des waypoints
        self.waypoint_radius = 2  # in meters
        # Polygon (work area) parameters
        self.polygon_points = [(-19.5, 19.5), (19.5, 19.5), (19.5, -19.5), (-19.5, -19.5)]  # in meters (rectangle containing the work area)
        
        self.window_scaling()
    
    def test_lowsampling(self):
        #################### DWA settings ####################
        self.time_step = 0.1  # [s] Time step for motion prediction
        self.time_horizon = 1 # [s] Prediction horizon for trajectory evaluation
        self.sampling_v_resolution = 0.05  # [m/s] Resolution for velocity sampling
        # DOIT ETRE INFERIEUR A L'ACCELERATION MAXIMALE FOIS LE TIME_STEP, SINON ON NE POURRA PAS ATTEINDRE CERTAINES VITESSES
        self.sampling_yaw_rate_resolution = 2.25 * math.pi / 180.0  # [rad/s] Resolution for yaw rate sampling
        # DOIT ETRE INFERIEUR A L'ACCELERATION ANGULAIRE MAXIMALE FOIS LE TIME_STEP, SINON ON NE POURRA PAS ATTEINDRE CERTAINES VITESSES ANGULAIRES

        # Cost function weights
        self.to_goal_cost_gain = 2.0  # Weight for goal-oriented cost
        self.speed_cost_gain = 0.5  # Weight for preferring higher speeds
        self.obstacle_cost_gain = 0.01  # Weight for obstacle avoidance
        self.close_cost_gain = 2.0 # Weight for goal closeness
        self.max_dist_for_cost = 50 # [m] distance maximale après laquelle le cout est le même

    def test_waypoint_segments(self):
        #################### Object settings ####################
        # Waypoint parameters
        self.waypoint_positions = [[(8.0, 2.0), (40.0, 8.0)], [(3.0, -6.0),(11.0, -46.0)], [(-5.0, -57.0), (2.0, -56.0)], [(12.0, -60.0), (14.0, -70.0)], [(17.0, -53.0), (50.0, -47.0)], [(-1.0, -6.0),(7.0, -46.0)], [(-30.0, -5.0), (-8.0, -1.0)], (self.init_x, self.init_y)]
            # in meters dans l'ordre des waypoints

    def test_boat_pov(self):
        #################### View settings ####################
        self.view_type = 'boat'  # 'static' or 'boat' (boat-centered view)
        self.zoom = 10  # Zoom factor for boat view (pixels per meter)
        self.window_scaling()

    def test_lac_mail(self):
        #################### Simulation settings ####################
        self.show_animation = True
        self.max_simulation_time = 100.0
        self.simulation_height = (-40, 60)  # in meters (= limite réelle minimale)
        self.simulation_width = (-40, 178) # in meters  (= limite réelle minimale)

        # Inital state
        self.init_x = -20.0 # in meters
        self.init_y = 0.0 # in meters
        self.init_yaw = np.radians(0) # in radians
        self.init_v = 0.0 # in m/s
        self.init_omega = 0.0 # in rad/s (=\dot{yaw})

        #################### DWA settings ####################
        self.sampling_v_resolution = 0.01  # [m/s] Resolution for velocity sampling
        self.sampling_yaw_rate_resolution = 1 * math.pi / 180.0  # [rad/s] Resolution for yaw rate sampling
        self.time_step = 0.1  # [s] Time step for motion prediction
        self.time_horizon = 0.5 # [s] Prediction horizon for trajectory evaluation

        #################### Object settings ####################
        # Buoy parameters
        self.buoy_positions = [(0, 0)] # in meters
        self.buoy_radius = 5  # in meters
        # Vehicle parameters
        self.vehicle_length = 1.0  # meters
        self.vehicle_width = 1.0   # meters
        self.vehicle_max_speed = 5.0  # [m/s] Maximum linear velocity
        self.vehicle_min_speed = -1  # [m/s] Minimum linear velocity (allows backward motion)
        self.vehicle_max_yaw_rate = 45.0 * math.pi / 180.0  # [rad/s] Maximum angular velocity
        self.vehicle_max_accel = 1  # [m/ss] Maximum linear acceleration
        self.vehicle_max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss] Maximum angular acceleration
        # Waypoint parameters
        self.waypoint_positions = [(160.0, 0.0), (self.init_x, self.init_y)]
            # in meters dans l'ordre des waypoints
        self.waypoint_radius = 2  # in meters
        # Starting zone parameters
        self.start_zone = ((-23, 3), (-17, -3)) # x_top_left_x, y_top_left_y, x_bottom_right_x, y_bottom_right_y  (in meters)
        # Polygon (work area) parameters
        self.polygon_points = [(-15, 30), (100, 35), (170, 45), (160, -20), (-10, -20), (-25, 0), (-25, 15)]  # in meters (rectangle containing the work area)

        #################### Pygame visualization settings ##################### 
        self.window_width = 1500 # pixels (= taille de la fenêtre)
        self.window_height = 800 # pixels (= taille de la fenêtre)
        self.border_offset_top = 0  # pixels (= margin around the simulation area)
        self.border_offset_left = 0  # pixels
        self.border_offset_right = 0  # pixels
        self.border_offset_bottom = 0  # pixels
        self.background_image_path = os.path.join("..","Images","Lac du mail.png")
        self.background_image_zero_position_local = (350, 570)  # pixels
        # Il faut ajuster cette position pour que l'origine (0,0) de la simulation corresponde au bon point sur l'image
        # L'origine (0,0) de la simulation est la bouée la plus proche du coin supérieur gauche de l'image
        self.background_image_scale_local = 431.1/54.10
        
        #################### View settings ####################
        self.view_type = 'static'  # 'static' or 'boat' (boat-centered view)
        self.zoom = 1  # Zoom factor for boat view (pixels per meter)

        self.window_scaling()
    
    def test_lac_mail_pov(self):
        self.test_lac_mail()

        #################### View settings ####################
        self.view_type = 'boat'  # 'static' or 'boat' (boat-centered view)
        self.zoom = 8  # Zoom factor for boat view (pixels per meter)

        self.window_scaling()

parameters = Parameters()
#parameters.test_affichage()
#parameters.test_lowsampling()
#parameters.test_boat_pov()
#parameters.test_waypoint_segments()
parameters.test_lac_mail()
#parameters.test_lac_mail_pov()
parameters.test_lowsampling()
