from plot_simulation import *
from simulation_parameters import parameters
import pygame

"""
Initialise pygames and define more high-level functions
"""

window_name = "DWA Simulation"

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((parameters.window_width, parameters.window_height))
pygame.display.set_caption(window_name)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

background_img, background_image_position = initialize_background()

def draw_basic_screen(screen, boat_x=None, boat_y=None, boat_theta=None):
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