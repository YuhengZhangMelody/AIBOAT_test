from src.path_planning_utils.simulation_parameters import parameters
import pygame
import numpy as np
import math

"""
This file contains plot functions to help reduce the number of lines in the main code
"""

def draw_background_boat_view(screen, background_img, boat_x, boat_y, boat_theta):
    """
    Draw background for boat-centric view.
    Rotate around the pivot point (background_image_zero_position_local * factor) so that
    world origin (0,0) always appears at reference_screen_pos.
    """
    if background_img is None:
        return
    
    # Get screen position of world origin (0, 0) which is the background reference point
    reference_screen_pos = parameters.get_screen_position(0, 0, boat_x, boat_y, boat_theta)

    # Pivot point inside the scaled background image (in pixels) that represents (0,0) of simulation
    pivot_x = parameters.background_image_zero_position_local[0] * parameters.factor
    pivot_y = parameters.background_image_zero_position_local[1] * parameters.factor

    # Rotate background image
    rotation_deg = 90 + math.degrees(-boat_theta)
    rotated_bg = pygame.transform.rotate(background_img, rotation_deg)
    rotated_w, rotated_h = rotated_bg.get_size()
    
    # Original image dimensions
    w, h = background_img.get_size()
    orig_center_x, orig_center_y = w / 2.0, h / 2.0
    
    # Vector from original image center to the pivot point
    vec_x = pivot_x - orig_center_x
    vec_y = pivot_y - orig_center_y
    
    # Rotate this vector by the same angle to find new offset from rotated image center
    angle_rad = math.radians(rotation_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Rotated offset vector
    rotated_vec_x = vec_x * cos_a - vec_y * sin_a
    rotated_vec_y = vec_x * sin_a + vec_y * cos_a
    
    # Rotated image center in screen coordinates, adjusted so pivot stays at reference_screen_pos
    rotated_center_x = reference_screen_pos[0] - rotated_vec_x
    rotated_center_y = reference_screen_pos[1] + rotated_vec_y
    
    # Calculate topleft and blit
    topleft_x = int(rotated_center_x - rotated_w / 2.0)
    topleft_y = int(rotated_center_y - rotated_h / 2.0)
    
    screen.blit(rotated_bg, (topleft_x, topleft_y))

def initialize_background() :
    """
    Load the background once pygame is initialized and shape it to the inner frame
    """

    if parameters.view_type == 'static':

        # Load and scale background image
        background_img = pygame.image.load(parameters.background_image_path).convert_alpha() # preserves transparency
        # Scale the background image to match the simulation scale
        original_size = background_img.get_size()
        new_size = (int(original_size[0] * parameters.factor), int(original_size[1] * parameters.factor))
        background_img = pygame.transform.scale(background_img, new_size)

        # Crop the image to fit within the simulation red rectangle
        red_top_left_x = parameters.center_offset[0] + parameters.simulation_width[0] * parameters.scale
        red_top_left_y = parameters.center_offset[1] + parameters.window_height - parameters.simulation_height[1] * parameters.scale
        red_width = (parameters.simulation_width[1] - parameters.simulation_width[0]) * parameters.scale
        red_height = (parameters.simulation_height[1] - parameters.simulation_height[0]) * parameters.scale
        crop_x_start = max(0, int(red_top_left_x - parameters.background_image_zero_position[0]))
        crop_y_start = max(0, int(red_top_left_y - parameters.background_image_zero_position[1]))
        crop_width = min(int(red_width), background_img.get_width() - crop_x_start)
        crop_height = min(int(red_height), background_img.get_height() - crop_y_start)
        background_img = background_img.subsurface((crop_x_start, crop_y_start, crop_width, crop_height))
        background_image_position = (parameters.background_image_zero_position[0] + crop_x_start, parameters.background_image_zero_position[1] + crop_y_start)
        return background_img, background_image_position
    else:
        # Load and scale background image
        background_img = pygame.image.load(parameters.background_image_path).convert_alpha() # preserves transparency
        # Scale the background image to match the simulation scale
        original_size = background_img.get_size()
        new_size = (int(original_size[0] * parameters.factor), int(original_size[1] * parameters.factor))
        background_img = pygame.transform.scale(background_img, new_size)
        background_image_position = (parameters.center_offset[0] + parameters.simulation_width[0] * parameters.scale, parameters.center_offset[1] + parameters.window_height - parameters.simulation_height[1] * parameters.scale)
        return background_img, background_image_position

def draw_starting_zone(screen, boat_x=None, boat_y=None, boat_theta=None):
    # Draw starting zone as green rectangle - only in static view
    if parameters.view_type == 'static':
        start_zone_top_left = (parameters.center_offset[0] + parameters.start_zone[0][0] * parameters.scale, parameters.center_offset[1] + parameters.window_height - (parameters.start_zone[0][1]) * parameters.scale)
        start_zone_bottom_right = (parameters.center_offset[0] + (parameters.start_zone[1][0]) * parameters.scale, parameters.center_offset[1] + parameters.window_height - parameters.start_zone[1][1] * parameters.scale)
        pygame.draw.rect(screen, parameters.GREEN, (start_zone_top_left[0], start_zone_top_left[1], start_zone_bottom_right[0] - start_zone_top_left[0], start_zone_bottom_right[1] - start_zone_top_left[1]))
    else:
        # Afficher la starting zone à partir de sa position à l'aide de get_screen_position pour qu'elle soit correctement positionnée dans la vue centrée sur le bateau
        # Le carré est crée sur une surface qui est ensuite affichée au bon endroit en calulant la rotation pour éviter les problèmes de rotation
        start_zone_center_x = (parameters.start_zone[0][0] + parameters.start_zone[1][0]) / 2.0
        start_zone_center_y = -(parameters.start_zone[0][1] + parameters.start_zone[1][1]) / 2.0
        start_zone_width = (parameters.start_zone[1][0] - parameters.start_zone[0][0]) * parameters.scale
        start_zone_height = -(parameters.start_zone[1][1] - parameters.start_zone[0][1]) * parameters.scale
        start_zone_surface = pygame.Surface((start_zone_width, start_zone_height), pygame.SRCALPHA)
        start_zone_surface.fill(parameters.GREEN)
        # Get screen position of the center of the starting zone
        start_zone_screen_pos = parameters.get_screen_position(start_zone_center_x, start_zone_center_y, boat_x, boat_y, boat_theta)
        # Rotate the starting zone surface by the negative of the boat's heading so it appears correctly oriented in the boat-centric view
        rotation_deg = math.degrees(-boat_theta)
        rotated_surface = pygame.transform.rotate(start_zone_surface, rotation_deg)
        rotated_w, rotated_h = rotated_surface.get_size()
        # Blit the rotated surface centered on the screen position of the starting zone
        # Calculer le topleft pour que le centre de la surface soit à start_zone_screen_pos
        topleft_x = int(start_zone_screen_pos[0] - rotated_w / 2.0)
        topleft_y = int(start_zone_screen_pos[1] - rotated_h / 2.0)
        screen.blit(rotated_surface, (topleft_x, topleft_y))

def draw_reference_path(screen, cx, cy, boat_x=None, boat_y=None, boat_theta=None, color=parameters.RED):
    # Draw reference path as red dots
    for i in range(len(cx)):
        screen_x, screen_y = parameters.get_screen_position(cx[i], cy[i], boat_x, boat_y, boat_theta)
        pygame.draw.circle(screen, color, (screen_x, screen_y), 2)

def draw_vehicle_trajectory(screen, x, y, boat_x=None, boat_y=None, boat_theta=None):
    # Draw vehicle trajectory as blue line
    if len(x) > 1:
        for i in range(1, len(x)):
            x1_screen, y1_screen = parameters.get_screen_position(x[i-1], y[i-1], boat_x, boat_y, boat_theta)
            x2_screen, y2_screen = parameters.get_screen_position(x[i], y[i], boat_x, boat_y, boat_theta)
            pygame.draw.line(screen, parameters.BLUE, (x1_screen, y1_screen), (x2_screen, y2_screen), 2)

def draw_current_target_point(screen, cx, cy, target_idx, boat_x=None, boat_y=None, boat_theta=None):
    # Draw current target point as green circle
    target_x, target_y = parameters.get_screen_position(cx[target_idx], cy[target_idx], boat_x, boat_y, boat_theta)
    pygame.draw.circle(screen, parameters.GREEN, (target_x, target_y), int(parameters.scale))

def draw_borders(screen, boat_x=None, boat_y=None, boat_theta=None):
    # Draw borders - only displayed in static view mode
    if parameters.view_type == 'static':
        top_left = (parameters.center_offset[0] + parameters.simulation_width[0] * parameters.scale, parameters.center_offset[1] + parameters.window_height - parameters.simulation_height[1] * parameters.scale)
        top_right = (parameters.center_offset[0] + parameters.simulation_width[1] * parameters.scale,  parameters.center_offset[1] + parameters.window_height - parameters.simulation_height[1] * parameters.scale)
        bottom_left = (parameters.center_offset[0] + parameters.simulation_width[0] * parameters.scale, parameters.center_offset[1] + parameters.window_height - parameters.simulation_height[0] * parameters.scale)
        bottom_right = (parameters.center_offset[0] + parameters.simulation_width[1] * parameters.scale, parameters.center_offset[1] + parameters.window_height - parameters.simulation_height[0] * parameters.scale)
        pygame.draw.polygon(screen, parameters.RED, [top_left, top_right, bottom_right, bottom_left], 2)
    else:
        # In boat-centric view, borders are not drawn since the view is centered on the boat
        pass

def draw_basic_screen(screen, background_img, background_image_position, boat_x=None, boat_y=None, boat_theta=None):
    """
    This function draw the basic requirements to print something using only the screen variable.
    In boat view mode, pass boat position and heading for proper print.
    """
    if parameters.view_type=="boat" and boat_x is not None and boat_y is not None and boat_theta is not None:
        draw_background_boat_view(screen, background_img, boat_x, boat_y, boat_theta)
    else:
        screen.blit(background_img, background_image_position) # Draw background image
    draw_borders(screen, boat_x, boat_y, boat_theta) # Draw borders of the screen
    draw_starting_zone(screen, boat_x, boat_y, boat_theta) # Draw starting zone as green rectangle