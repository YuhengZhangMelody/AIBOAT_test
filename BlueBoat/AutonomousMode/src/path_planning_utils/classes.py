from src.path_planning_utils.simulation_parameters import parameters
import pygame
import math
import numpy as np


class Polygon:
    """
    Représente un polygone convexe ou non défini par des sommets ordonnés.
    Utilisé pour définir une zone de travail et tester si un Boat s'y trouve.
    """
    
    def __init__(self, points, color=None):
        """
        Initialise le polygone avec une liste ordonnée de points.
        
        Args:
            points: Liste de tuples (x, y) représentant les sommets du polygone
            color: Couleur pour l'affichage (par défaut, vert semi-transparent)
        """
        self.points = points
        self.color = color if color is not None else parameters.DARK_RED
        self.alpha = 0  # Transparence pour affichage (0->transparent / 255->opaque)
    
    def is_point_inside(self, x, y):
        """
        Teste si un point (x, y) est à l'intérieur du polygone.
        Utilise l'algorithme ray casting.
        
        Args:
            x: Coordonnée x du point
            y: Coordonnée y du point
        
        Returns:
            True si le point est à l'intérieur, False sinon
        """
        n = len(self.points)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def is_boat_inside(self, boat):
        """
        Teste si un bateau (rectangle) est entièrement à l'intérieur du polygone.
        
        Args:
            boat: Objet Boat
        
        Returns:
            True si tous les points du bateau sont à l'intérieur du polygone, False sinon
        """
        # Teste le centre du bateau
        if not self.is_point_inside(boat.x, boat.y):
            return False
        
        # Teste tous les coins du bateau
        corners = boat.get_corners()
        for corner in corners:
            if not self.is_point_inside(corner[0], corner[1]):
                return False
        
        return True
    
    def draw(self, screen, boat_x=None, boat_y=None, boat_theta=None):
        """
        Affiche le polygone sur l'écran Pygame.
        
        Args:
            screen: Surface Pygame où afficher le polygone
            boat_x: Position x du bateau (optionnel, pour mode bateau-centré)
            boat_y: Position y du bateau (optionnel, pour mode bateau-centré)
            boat_theta: Orientation du bateau (optionnel, pour rotation du monde)
        """
        if len(self.points) < 2:
            return
        
        # Convertir les points en coordonnées écran
        screen_points = []
        for x, y in self.points:
            screen_x, screen_y = parameters.get_screen_position(x, y, boat_x, boat_y, boat_theta)
            screen_points.append((screen_x, screen_y))
        
        # Créer une surface avec transparence pour le polygone
        if len(screen_points) >= 3:
            # Créer une surface temporaire pour le remplissage transparent
            temp_surface = pygame.Surface((parameters.window_width, parameters.window_height), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surface, (*self.color, self.alpha), screen_points)
            screen.blit(temp_surface, (0, 0))
        
        # Dessiner le contour du polygone
        pygame.draw.lines(screen, self.color, True, screen_points, 3)


class Buoy:
    """
    Représente une bouée obstacle dans l'environnement
    Utilisé pour la détection de collision
    """
    
    def __init__(self, x, y, radius=None, color=None):
        self.x = x
        self.y = y
        self.radius = radius if radius is not None else parameters.buoy_radius
        self.color = color if color is not None else parameters.ORANGE
        self.collision = False  # Flag pour visualisation collision
        self.tolerance = 3
    
    def draw(self, screen, boat_x=None, boat_y=None, boat_theta=None):
        """Affichage de la bouée
        
        Args:
            screen: Surface Pygame
            boat_x: Position x du bateau (optionnel, pour mode bateau-centré)
            boat_y: Position y du bateau (optionnel, pour mode bateau-centré)
            boat_theta: Orientation du bateau (optionnel, pour rotation du monde)
        """
        current_color = parameters.DARK_RED if self.collision else self.color

        buoy_screen_x, buoy_screen_y = parameters.get_screen_position(self.x, self.y, boat_x, boat_y, boat_theta)
        
        # Cercle principal
        pygame.draw.circle(screen, current_color, (buoy_screen_x, buoy_screen_y), int(self.radius * parameters.scale))
        pygame.draw.circle(screen, parameters.BLACK, (buoy_screen_x, buoy_screen_y), int(self.radius * parameters.scale), 2)
        
        # Point central
        pygame.draw.circle(screen, parameters.WHITE, (buoy_screen_x, buoy_screen_y), int(0.6*parameters.scale))
        
        # Cercle de tolerance (= où une collision est détectée)
        if self.tolerance > 0:
            pygame.draw.circle(screen, parameters.ORANGE, (buoy_screen_x, buoy_screen_y), int((self.radius + self.tolerance) * parameters.scale), 1)
        
    def check_collision_with_boat(self, boat):
        """
        Détection de collision entre le waypoint (cercle) et le bateau (rectangle)
        Retourne True si collision détectée
        """
        # Test rapide: distance entre centres
        dx = self.x - boat.x
        dy = self.y - boat.y
        distance_center = math.sqrt(dx**2 + dy**2)
        if distance_center < self.radius + boat.width / 2 + self.tolerance:
            return True
        
        # Test précis: distance aux coins du bateau
        corners = boat.get_corners()
        for corner in corners:
            dx = self.x - corner[0]
            dy = self.y - corner[1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < self.radius + self.tolerance:
                return True
    
        return False
    
    def get_position(self):
        """Retourne (x, y, radius) pour intégration path planning"""
        return (self.x, self.y, self.radius)


class Waypoint:
    """
    Représente un point de navigation (START ou GOAL)
    Ou une ligne de navigation
    Utilisé pour définir les objectifs de navigation
    """
    
    def __init__(self, x, y, waypoint_type='goal', radius=2, waypoint_shape='point', number=None, x2= None, y2=None):
        self.x = x
        self.y = y
        self.radius = radius # Dans le cas d'un segment, il s'agit de l'épaisseur
        self.type = waypoint_type  # 'start' ou 'goal'
        self.reached = False
        self.shape = waypoint_shape # 'point' ou 'segment'
        self.number = number if number is not None else 0
        # Deuxième point si type == segment
        self.x2 = x2
        self.y2 = y2
        
    def draw(self, screen, font, boat_x=None, boat_y=None, boat_theta=None):
        """Affichage du waypoint avec label
        
        Args:
            screen: Surface Pygame
            font: Police Pygame
            boat_x: Position x du bateau (optionnel, pour mode bateau-centré)
            boat_y: Position y du bateau (optionnel, pour mode bateau-centré)
            boat_theta: Orientation du bateau (optionnel, pour rotation du monde)
        """
        if self.type == 'start':
            color = parameters.GREEN
            text = 'S'
            inner_color = (0, 200, 0)
            text_color = parameters.BLACK
        else:
            color = parameters.RED if not self.reached else parameters.YELLOW
            text = str(self.number)
            inner_color = (200, 0, 0) if not self.reached else (0, 200, 200)
            # Texte adapté : noir sur couleurs claires
            text_color = parameters.BLACK
        
        ax_p, ay_p = parameters.get_screen_position(self.x, self.y, boat_x, boat_y, boat_theta)

        # Label avec couleur adaptée
        text_surface = font.render(text, True, text_color).convert_alpha()
        txt_w, txt_h = text_surface.get_size()
        text_surface = pygame.transform.smoothscale(text_surface, (int((self.radius*parameters.scale) * txt_w//txt_h), int(self.radius*parameters.scale)))

        if self.shape == 'segment' and self.x2 is not None and self.y2 is not None:
            bx_p, by_p = parameters.get_screen_position(self.x2, self.y2, boat_x, boat_y, boat_theta)

            # Ligne épaisse avec bordure noire
            pygame.draw.line(screen, parameters.BLACK, (ax_p, ay_p), (bx_p, by_p), int(self.radius * parameters.scale + 6)) # Bordure noire  
            pygame.draw.line(screen, color, (ax_p, ay_p), (bx_p, by_p), int(self.radius * parameters.scale)) # Ligne colorée
            pygame.draw.line(screen, inner_color, (ax_p, ay_p), (bx_p, by_p), int((self.radius - 1) * parameters.scale)) # Ligne couleur intérieure

            # Texte centré sur le milieu du segment
            text_rect = text_surface.get_rect(center=((ax_p + bx_p) // 2, (ay_p + by_p) // 2))
                      

        else:
            # Cercles concentriques
            pygame.draw.circle(screen, color, (ax_p, ay_p), int(self.radius*parameters.scale))
            pygame.draw.circle(screen, inner_color, (ax_p, ay_p), int((self.radius - 1)*parameters.scale))
            pygame.draw.circle(screen, parameters.BLACK, (ax_p, ay_p), int(self.radius*parameters.scale), 3)
            
            # Texte centré sur le waypoint
            text_rect = text_surface.get_rect(center=(ax_p, ay_p))

        screen.blit(text_surface, text_rect)
    
    def get_distance(self, boat):
        """Distance euclidienne au bateau"""
        if self.shape == 'segment' and self.x2 is not None and self.y2 is not None:
            # Calcul de la distance d'un point à un segment
            px = self.x2 - self.x
            py = self.y2 - self.y
            norm = px*px + py*py
            u = ((boat.x - self.x) * px + (boat.y - self.y) * py) / norm if norm > 0 else 0
            u = max(0, min(1, u))  # Clamp u to [0, 1]
            closest_x = self.x + u * px
            closest_y = self.y + u * py
            dx = boat.x - closest_x
            dy = boat.y - closest_y
            return math.sqrt(dx**2 + dy**2)
        else:
            dx = self.x - boat.x
            dy = self.y - boat.y
            return math.sqrt(dx**2 + dy**2)
    
    def get_position(self, boat=None):
        """Retourne (x, y) pour intégration path planning"""
        """Retourne la coordonnée x du waypoint"""
        if self.shape == 'segment' and self.x2 is not None and self.y2 is not None and boat is not None:
            # Retourne les coordonnées x et y du point le plus proche sur le segment
            px = self.x2 - self.x
            py = self.y2 - self.y
            # Calcul de la projection du bateau sur le segment pour trouver le point le plus proche
            norm = px*px + py*py
            # Avoid division by zero if the segment is a point
            u = ((boat.x - self.x) * px + (boat.y - self.y) * py) / norm if norm > 0 else 0
            # u est la position du point projeté sur le segment, exprimée en fraction de la longueur du segment.
            # Clamp u to [0, 1] to stay within the segment
            u = max(0, min(1, u))  # Clamp u to [0, 1]
            closest_x = self.x + u * px
            closest_y = self.y + u * py
            return closest_x, closest_y
        else:
            return self.x, self.y # Attention, pour les segments, on retourne la position x et y le plus proche sur le segment, pas le centre du segment
    
    
    def check_collision_with_boat(self, boat, tolerance = 0):
        """
        Détection de collision cercle-rectangle entre ce waypoint (cercle)
        et le bateau (rectangle), avec tolérance optionnelle en mètres.
        Retourne True si collision détectée ou si distance <= tolerance.
        
        Args:
            boat: objet bateau (doit fournir x, y, width, get_corners())
            tolerance: distance supplémentaire de tolérance en mètres (défaut: 0)
        """
        if self.shape == 'segment' and self.x2 is not None and self.y2 is not None:
            # Pour un segment, on considère une "zone de collision" rectangulaire autour du segment
            # On peut faire un test plus précis en calculant la distance du bateau au segment et en comparant à radius + tolerance
            # Calcul de la distance d'un point à un segment
            px = self.x2 - self.x
            py = self.y2 - self.y
            norm = px*px + py*py
            u = ((boat.x - self.x) * px + (boat.y - self.y) * py) / norm if norm > 0 else 0
            u = max(0, min(1, u))  # Clamp u to [0, 1]
            closest_x = self.x + u * px
            closest_y = self.y + u * py
            
            dx = boat.x - closest_x
            dy = boat.y - closest_y
            distance_to_segment = math.sqrt(dx**2 + dy**2)
            
            if distance_to_segment < self.radius + tolerance:
                return True
        else:
            # Test rapide: distance entre centres
            dx = self.x - boat.x
            dy = self.y - boat.y
            distance_center = math.sqrt(dx**2 + dy**2)
            if distance_center < self.radius + boat.width / 2 + tolerance:
                return True

            # Test précis: distance aux coins du bateau
            corners = boat.get_corners()
            for corner in corners:
                dx = self.x - corner[0]
                dy = self.y - corner[1]
                distance = math.sqrt(dx**2 + dy**2)
                if distance < self.radius + tolerance:
                    return True

            return False

class Boat:
    """
    Implemente un modèle de bateau
    Utilisé pour la simulation et le contrôle 
    """
    
    def __init__(self, x, y, theta, alpha=255):
        # État courant
        self.x = x
        self.y = y
        self.theta = theta  # Orientation en radians
        
        # État initial (pour reset)
        self.initial_x = x
        self.initial_y = y
        self.initial_theta = theta
        
        # Vitesses du robot
        self.v = 0.0      # Vitesse linéaire
        self.omega = 0.0  # Vitesse angulaire
        
        # Dimensions pour affichage
        self.length = parameters.vehicle_length
        self.width = parameters.vehicle_width
        self.alpha = alpha  # Transparence pour affichage (0->transparent / 255->opaque)
        
        # Trajectoire
        self.trail = []
        self.max_trail_length = 300
        
        # Métriques
        self.collision_count = 0
        self.total_distance = 0.0

        # État réel pour SLAM (non utilisé pour contrôle, mais utile pour visualisation)
        self.x_true = x
        self.y_true = y
        self.theta_true = theta  # Orientation en radians
        
        # Paramètre d'observation pour SLAM
        self.RANGE_YAW_DIFFERENCE = (-25, 25) # En DEGRES
        self.MAX_RANGE = 40.0  # maximum observation range
    
    @staticmethod
    def motion_model_np(x, u, dt):
        """
        Static motion model compatible with EKF usage.
        x: numpy array shape (3,1) or (3,) -> [x, y, theta]
        u: numpy-like control [v, omega] # unité m/s et rad/s
        dt: time step
        Retourne xp (3x1 numpy array)
        """
        # Ensure numpy arrays
        x_arr = np.array(x).reshape((3, 1))
        u_arr = np.array(u).reshape((-1, 1))

        xp = np.array([[x_arr[0, 0] + u_arr[0, 0] * dt * math.cos(x_arr[2, 0])],
                       [x_arr[1, 0] + u_arr[0, 0] * dt * math.sin(x_arr[2, 0])],
                       [x_arr[2, 0] + u_arr[1, 0] * dt]])

        xp[2, 0] = (xp[2, 0] + math.pi) % (2 * math.pi) - math.pi # Normalisation angle entre -pi et pi

        return xp.reshape((3, 1))
    
    def reset_to_start(self):
        """Réinitialise le bateau à sa position initiale"""
        self.x = self.initial_x
        self.y = self.initial_y
        self.theta = self.initial_theta
        self.trail = []
        self.collision_count = 0
        self.total_distance = 0.0
        self.omega_left = 0.0
        self.omega_right = 0.0
        self.v = 0.0
        self.omega = 0.0
    
    def get_corners(self):
        """Calcule les 4 coins du rectangle représentant le bateau"""
        corners = [
            (-self.length / 2, -self.width / 2),
            (self.length / 2, -self.width / 2),
            (self.length / 2, self.width / 2),
            (-self.length / 2, self.width / 2)
        ]
        
        # Rotation + translation
        rotated_corners = []
        for corner in corners:
            x_rot = corner[0] * math.cos(self.theta) - corner[1] * math.sin(self.theta)
            y_rot = corner[0] * math.sin(self.theta) + corner[1] * math.cos(self.theta)
            rotated_corners.append((self.x + x_rot, self.y + y_rot))
        
        return rotated_corners
    
    def show(self, screen):
        """
        Draw the boat as a blue rectangle on the Pygame screen.

        The boat is represented as a rectangle rotated according to its theta angle in static view.
        In 'boat' view mode, the boat is always at the center pointing upward (no rotation).
        
        Args:
            screen: Pygame screen to draw on.
        """
        # Créer une surface temporaire pour le remplissage transparent
        temp_surface = pygame.Surface((parameters.window_width, parameters.window_height), pygame.SRCALPHA)

        # Convert to pixels
        half_length = parameters.vehicle_length / 2
        half_width = parameters.vehicle_width / 2
        # Define rectangle corners relative to center
        points = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        if parameters.view_type == 'boat':
            # Boat-centric view: boat is always at center pointing upward (no rotation)
            rotated_points = []
            cos_t = math.cos(np.radians(-90))  # No rotation
            sin_t = math.sin(np.radians(-90))
            for px, py in points:
                # Rotate local corner into world frame
                x_rot = px * cos_t - py * sin_t
                y_rot = px * sin_t + py * cos_t
                world_x = self.x + x_rot
                world_y = self.y + y_rot
                screen_x, screen_y = parameters.get_screen_position(world_x, world_y, self.x, self.y)
                rotated_points.append((int(screen_x), int(screen_y)))
        else:
            # Static view: boat rotates according to theta
            # Static view: compute each corner in world coordinates using canonical rotation
            # then convert to screen coords via parameters.get_screen_position() to keep
            # transforms consistent with other objects.
            rotated_points = []
            cos_t = math.cos(self.theta)
            sin_t = math.sin(self.theta)
            for px, py in points:
                # Rotate local corner into world frame
                x_rot = px * cos_t - py * sin_t
                y_rot = px * sin_t + py * cos_t
                world_x = self.x + x_rot
                world_y = self.y + y_rot
                screen_x, screen_y = parameters.get_screen_position(world_x, world_y)
                rotated_points.append((int(screen_x), int(screen_y)))
                    
        pygame.draw.polygon(temp_surface, (*parameters.WHITE, self.alpha), rotated_points)
        screen.blit(temp_surface, (0, 0))

    def show_range(self, screen):
        if parameters.realtime:
            angle_debut = self.theta +  self.RANGE_YAW_DIFFERENCE[0] *math.pi/180 # En radian
            angle_fin = self.theta +  self.RANGE_YAW_DIFFERENCE[1] *math.pi/180 # En radian
        else:
            angle_debut = self.theta_true +  self.RANGE_YAW_DIFFERENCE[0] *math.pi/180 # En radian
            angle_fin = self.theta_true +  self.RANGE_YAW_DIFFERENCE[1] *math.pi/180 # En radian

        pie = pygame.Surface((parameters.window_width, parameters.window_height), pygame.SRCALPHA)

        # Start list of polygon points
        if parameters.realtime:
            p = [parameters.get_screen_position(self.x, self.y, self.x, self.y, self.theta)]
        else:
            p = [parameters.get_screen_position(self.x_true, self.y_true, self.x, self.y, self.theta)]

        # Get points on arc
        for n in np.linspace(angle_debut, angle_fin, num=50):
            x =  self.MAX_RANGE*math.cos(n) 
            y =  self.MAX_RANGE*math.sin(n) # Negatif parce ce que l'affichage vertical est inversé
            if parameters.view_type == 'boat':
                if parameters.realtime:
                    screen_x, screen_y = parameters.get_screen_position(self.x + x, self.y + y, self.x, self.y, self.theta)
                else:
                    screen_x, screen_y = parameters.get_screen_position(self.x_true + x, self.y_true + y, self.x, self.y, self.theta)
            else:
                if parameters.realtime:
                    screen_x, screen_y = parameters.get_screen_position(self.x + x, self.y + y)
                else:
                    screen_x, screen_y = parameters.get_screen_position(self.x_true + x, self.y_true + y)
            p.append((screen_x, screen_y))
        p.append(p[0])
    
        # Draw pie segment
        if len(p) > 2:
            pygame.draw.polygon(pie, (255, 255, 0, 64), p)
        screen.blit(pie, (0, 0))

    def show_true(self, screen):
        """
        Draw the boat as a blue rectangle on the Pygame screen.

        The boat is represented as a rectangle rotated according to its theta angle in static view.
        In 'boat' view mode, the boat is always at the center pointing upward (no rotation).
        
        Args:
            screen: Pygame screen to draw on.
        """
        # Créer une surface temporaire pour le remplissage transparent
        temp_surface = pygame.Surface((parameters.window_width, parameters.window_height), pygame.SRCALPHA)
        temp_surface2 = pygame.Surface((parameters.window_width, parameters.window_height), pygame.SRCALPHA)

        # Convert to pixels
        half_length = parameters.vehicle_length / 2
        half_width = parameters.vehicle_width / 2
        # Define rectangle corners relative to center
        points = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        if parameters.view_type == 'boat':
            # Boat is at the center but not true_boat. So we update the points with the true boat position and orientation, then convert to screen coords.
            rotated_points = []
            cos_t = math.cos(self.theta_true)
            sin_t = math.sin(self.theta_true)
            for px, py in points:
                # Rotate local corner into world frame
                x_rot = px * cos_t - py * sin_t
                y_rot = px * sin_t + py * cos_t
                world_x = self.x_true + x_rot
                world_y = self.y_true + y_rot
                screen_x, screen_y = parameters.get_screen_position(world_x, world_y, self.x, self.y, self.theta)
                rotated_points.append((int(screen_x), int(screen_y)))
    
        else:
            # Static view: boat rotates according to theta
            # Static view: compute each corner in world coordinates using canonical rotation
            # then convert to screen coords via parameters.get_screen_position() to keep
            # transforms consistent with other objects.
            rotated_points = []
            cos_t = math.cos(self.theta_true)
            sin_t = math.sin(self.theta_true)
            for px, py in points:
                # Rotate local corner into world frame
                x_rot = px * cos_t - py * sin_t
                y_rot = px * sin_t + py * cos_t
                world_x = self.x_true + x_rot
                world_y = self.y_true + y_rot
                screen_x, screen_y = parameters.get_screen_position(world_x, world_y)
                rotated_points.append((int(screen_x), int(screen_y)))
                    
        pygame.draw.polygon(temp_surface, (*parameters.WHITE, 255), rotated_points) # Pas de transparence pour le vrai bateau
        screen.blit(temp_surface, (0, 0))
        screen.blit(temp_surface2, (0, 0))
    
    def display_state(self, screen, time, font):
        """
        Display the current state (speed, position, yaw, time) on screen.
        
        Args:
            screen: Pygame screen to draw on.
            time: Current simulation time.
            font: Pygame font to use for rendering text.
        """
        speed_text = font.render(f"Speed: {self.v * 3.6:.2f} km/h", True, parameters.BLACK)
        screen.blit(speed_text, (10, 10))
        speed_text = font.render(f"Speed: {self.v:.2f} m/s", True, parameters.BLACK)
        screen.blit(speed_text, (10, 30))
        x_text = font.render(f"X: {self.x:.2f} m", True, parameters.BLACK)
        screen.blit(x_text, (10, 50))
        y_text = font.render(f"Y: {self.y:.2f} m", True, parameters.BLACK)
        screen.blit(y_text, (10, 70))
        yaw_text = font.render(f"Yaw: {self.theta * 180 / math.pi:.2f} °", True, parameters.BLACK)
        screen.blit(yaw_text, (10, 90))
        time_text = font.render(f"Time: {time:.0f} s", True, parameters.BLACK)
        screen.blit(time_text, (10, 110))
    
    def draw_arrow(self, screen):
        """
        Draw an arrow indicating the boat's heading direction.
        In boat view, arrow is fixed pointing upward.

        Args:
            screen: Pygame screen to draw on.
        """
        if parameters.view_type == 'boat':
            # Boat-centric view: arrow always points upward
            start_x = int(parameters.center_offset[0])
            start_y = int(parameters.center_offset[1])
            end_x = int(parameters.center_offset[0])
            end_y = int(parameters.center_offset[1] - self.length * parameters.scale)
        else:
            # Static view: arrow follows boat heading. Compute end point in world
            # coordinates then convert to screen to respect global transforms.
            start_x, start_y = parameters.get_screen_position(self.x, self.y)
            end_world_x = self.x + self.length * math.cos(self.theta)
            end_world_y = self.y + self.length * math.sin(self.theta)
            end_x, end_y = parameters.get_screen_position(end_world_x, end_world_y)
        
        pygame.draw.line(screen, (0, 0, 0), (start_x, start_y), (end_x, end_y), 2)
        
    def get_state(self):
        """Retourne l'état complet (utile pour RL)"""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'v': self.v,
            'omega': self.omega,
            'omega_left': self.omega_left,
            'omega_right': self.omega_right,
            'collision_count': self.collision_count,
            'total_distance': self.total_distance
        }
