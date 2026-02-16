# ui_map.py
import math
import pygame
from boat_state import ThreadSafeBoatState

from ui.ui_utils import _world_to_screen, draw_fov_cone, draw_heading_line, _boat_to_world, _cam_to_boat

class UIVision:
    """
    Equivalent pygame de ton widget PySide6 Map.
    - Lit les variables via boat_state (tu ajusteras les clés comme tu veux).
    - API standardisée: update() + render(screen).
    """

    def __init__(self, boat_state: ThreadSafeBoatState, window_size):
        self.boat_state = boat_state

        self.window_size = window_size
        self.screen_width = window_size[0]
        self.screen_height = window_size[1]

        # --- paramètres par défaut (peuvent être surchargés par boat_state) ---
        self.scale = 40               # pixels par unité monde
        self.fov_deg = 60.0           # ouverture du cône (degrés)
        self.fov_range = 5.0          # portée (unités monde)

        # style
        self.bg_color = (255, 255, 255)
        self.origin_color = (0, 0, 0)
        self.point_color = (255, 0, 0)
        self.text_color = (0, 0, 0)
        self.fov_color = (0, 120, 255, 80)  # RGBA (alpha)

        # font (pygame doit être init avant ou le module font doit être init)
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont(None, 18)

        # surface pygame transparente dédiée pour l'alpha (FOV)
        self._fov_surface = pygame.Surface(window_size, pygame.SRCALPHA)

    
    # ----------- standard API -----------
    def update(self):
        """
        Ici :
        - lire des paramètres dynamiques
        - faire des filtres / moyennes
        - gérer des animations
        """

    def render(self, screen: pygame.Surface):
        """
        Dessine:
        - fond
        - cône FOV semi-transparent
        - origine
        - points + distance
        """

        # ---- lecture état ----
        # ⚠️ idés: tu adapteras les clés côté boat_state
        heading = self.boat_state["heading"]  # radians, 0 = +x, CCW positif
        if heading is None:
            heading = 0.0
        heading = float(heading)

        # points: [(x, y, dist), ...] en repère "capteur" (comme ton widget)
        points = self.boat_state["detected_objects"]
        if points is None:
            points = []
        # IMPORTANT: si map_points est une liste mutable stockée dans boat_state,
        # assure-toi de donner une copie (snapshot) côté boat_state. Ici, on se protège un minimum:
        points = list(points)

        # ---- fond ----
        screen.fill(self.bg_color)

        # origine (0,0) au centre

        origin = _world_to_screen(self.scale, 0.0, 0.0, screen)
        pygame.draw.circle(screen, self.origin_color, (int(origin[0]), int(origin[1])), 3)

        draw_heading_line(
            screen,
            heading_rad=heading,
            length_px=100.0,
            origin_px=origin,
            color=(255, 0, 0)
        )

        # ---- FOV pour chaque caméra ----
        for cam in self.boat_state["Cameras"].values():

            # position caméra en repère monde
            try :
                Xcam_world, Ycam_world = _boat_to_world(
                    cam["XCam"],
                    cam["YCam"],
                    heading
                )

            # position caméra en repère écran
            
                Xcam_screen, Ycam_screen = _world_to_screen(
                    self.scale,
                    Xcam_world,
                    Ycam_world,
                    screen
                )

                Yawcam_rad = math.radians(cam["YawCam_deg"])
                
                self._fov_surface = draw_fov_cone(
                    screen,
                    heading_rad=Yawcam_rad+heading,
                    fov_range_world=self.fov_range,
                    fov_deg=cam["FovCam_deg"],
                    scale_px_per_world=self.scale,
                    origin_px=(
                        Xcam_screen,
                        Ycam_screen,
                    ),
                    work_surface=self._fov_surface,   # réutilisation perf
                )
            
            except Exception as e:

                print("Erreur à l'affichage")

        for obj in points:

            #position objet en repère bateau
            Xobj_boat, Yobj_boat = _cam_to_boat(
                obj['x'],
                obj['y'],
                self.boat_state["Cameras"][obj["IdCam"]]["XCam"],
                self.boat_state["Cameras"][obj["IdCam"]]["YCam"],
                math.radians(self.boat_state["Cameras"][obj["IdCam"]]["YawCam_deg"])
            )
            #position objet en repère monde
            Xobj_world, Yobj_world = _boat_to_world(
                Xobj_boat,
                Yobj_boat,
                heading
            )

            # position objet en repère écran
            Xobj_screen, Yobj_screen = _world_to_screen(
                self.scale,
                Xobj_world,
                Yobj_world,
                screen
            )

            # dessiner point + distance
            try :
                pygame.draw.circle(screen, self.point_color, (int(Xobj_screen), int(Yobj_screen)), 3)

            except Exception as e: 
                print(f"Erreur à l'affichage : {e}")

            label = f"{float(obj['dist']):.2f}"
            text = self.font.render(label, True, self.text_color)
            screen.blit(text, (int(Xobj_screen) + 6, int(Yobj_screen) - 16))

        
