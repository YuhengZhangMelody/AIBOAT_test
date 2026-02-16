import math
import pygame

# ----------- helpers -----------
def _world_to_screen(scale, x, y, screen):
    w, h = screen.get_size()
    sx = w / 2 + x * scale
    sy = h / 2 - y * scale
    return sx, sy

def _boat_to_world(x,y, heading_rad):
    # même formule que ton code PySide
    X = (x * math.cos(heading_rad) - y * math.sin(heading_rad))
    Y = (x * math.sin(heading_rad) + y * math.cos(heading_rad))
    return X, Y

def _cam_to_boat(x,y,Xcam_boat, Ycam_boat, YawCam_rad):
    # Convert camera position from boat frame to world frame
    Xcam_world = (x * math.cos(YawCam_rad) - y * math.sin(YawCam_rad))+Xcam_boat
    Ycam_world = (x * math.sin(YawCam_rad) + y * math.cos(YawCam_rad))+Ycam_boat
    return Xcam_world, Ycam_world

def _rotate_xy(x: float, y: float, yaw: float):
    # même formule que ton code PySide
    X = (x * math.cos(yaw) + y * math.sin(yaw))
    Y = (y * math.cos(yaw) - x * math.sin(yaw))
    return X, Y

def _sector_points(self, center, radius_px, start_deg, end_deg, step_deg=3.0):
    """
    Construit un polygone (liste de points) représentant un secteur circulaire.
    Angles en degrés, repère écran pygame: +x à droite, +y en bas.
    """
    cx, cy = center
    pts = [(cx, cy)]
    if end_deg < start_deg:
        end_deg += 360.0

    a = start_deg
    while a <= end_deg + 1e-6:
        rad = math.radians(a)
        x = cx + radius_px * math.cos(rad)
        y = cy - radius_px * math.sin(rad)  # signe - pour garder le CCW "math" vers le haut écran
        pts.append((x, y))
        a += step_deg

    # dernier point exact
    rad = math.radians(end_deg)
    x = cx + radius_px * math.cos(rad)
    y = cy - radius_px * math.sin(rad)
    pts.append((x, y))

    return pts

def draw_fov_cone(
    screen: pygame.Surface,
    heading_rad: float,
    fov_range_world: float,
    fov_deg: float,
    scale_px_per_world: float,
    *,
    origin_px: tuple[float, float] | None = None,
    color_rgba: tuple[int, int, int, int] = (0, 120, 255, 80),
    step_rad: float = 0.1,
    work_surface: pygame.Surface | None = None,
) -> pygame.Surface:
    """
    Dessine un cône de vision semi-transparent (secteur circulaire) sur `screen`.

    Paramètres:
    - heading_rad: cap en radians (0 = +x, CCW positif) dans le repère "math".
    - fov_range_world: portée du cône en unités monde.
    - fov_deg: ouverture du cône en degrés.
    - scale_px_per_world: conversion unités monde -> pixels.
    - origin_px: (ox, oy) en pixels. Par défaut: centre de l'écran.
    - color_rgba: couleur du cône (R,G,B,A).
    - step_deg: pas angulaire pour approximer l'arc (plus petit = plus lisse).
    - work_surface: surface alpha réutilisable (perf). Si None, créée et renvoyée.

    Retour:
    - la surface alpha utilisée (à réutiliser d'une frame à l'autre).
    """
    if origin_px is None:
        w, h = screen.get_size()
        origin_px = (w / 2.0, h / 2.0)

    ox, oy = origin_px
    radius_px = float(fov_range_world) * float(scale_px_per_world)

    # Surface alpha de travail (calque)
    if work_surface is None or work_surface.get_size() != screen.get_size():
        work_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    work_surface.fill((0, 0, 0, 0))  # clear transparent

    # Conversion heading -> angle écran
    # On garde la même logique que ton PySide:
    # heading_deg = degrees(heading)
    # center_deg_screen = -heading_deg
    # start = center - fov/2 ; end = start + fov

    fov_rad = math.radians(fov_deg)

    center_rad_screen = heading_rad
    start_rad = center_rad_screen - float(fov_rad) / 2.0
    end_rad = start_rad + float(fov_rad)

    # Construit le polygone du secteur
    pts = [(ox, oy)]
    if end_rad < start_rad:
        end_rad += 2*math.pi

    a = start_rad
    while a <= end_rad + 1e-6:
        x = ox + radius_px * math.cos(a)
        y = oy - radius_px * math.sin(a)  # '-' pour repère écran (y vers le bas)
        pts.append((x, y))
        a += float(step_rad)

    # dernier point exact (pour fermer proprement)
    x = ox + radius_px * math.cos(a)
    y = oy - radius_px * math.sin(a)
    pts.append((x, y))

    pygame.draw.polygon(work_surface, color_rgba, pts)
    screen.blit(work_surface, (0, 0))

    return work_surface


def draw_heading_line(
    screen: pygame.Surface,
    heading_rad: float,
    length_px: float,
    *,
    origin_px: tuple[float, float] | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
) -> None:
    """
    Dessine une demi-droite (rayon) partant de l'origine et indiquant le heading.

    Convention:
    - heading_rad en radians
    - 0 rad = +x (droite écran)
    - CCW positif
    - y écran vers le bas (d'où le '-' sur sin)
    """
    if origin_px is None:
        w, h = screen.get_size()
        origin_px = (w / 2.0, h / 2.0)

    ox, oy = origin_px

    hx = ox + float(length_px) * math.cos(float(heading_rad))
    hy = oy - float(length_px) * math.sin(float(heading_rad))

    pygame.draw.line(
        screen,
        color,
        (int(ox), int(oy)),
        (int(hx), int(hy)),
        int(width),
    )