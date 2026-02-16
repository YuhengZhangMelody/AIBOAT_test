import pygame
from boat_state import ThreadSafeBoatState
from ui.ui_vision import UIVision
from ui.ui_path_planning import UIPathPlanning

UI_VISION = 0
UI_PP = 1

# Dimensions de l'écran
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200


class UISystem:
    """
        Cette classe est responsable de l'initialisation de pygame,
        elle contient aussi plusieurs types d'interface (toutes dans ui/*.py)
        et affiche une interface plutôt qu'une autre selon boat_state["current_ui"]
    """

    # nombre d'appels à la fonction "step" chaque secondes
    LOOP_SPEED_HZ = 60

    def __init__(self, boat_state: ThreadSafeBoatState):
        
        pygame.init()

        self.boat_state = boat_state
        
        self.ui_vision = UIVision(boat_state, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.ui_pp = UIPathPlanning(boat_state, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.selected_ui = 1 # UI_WELCOME_SCREEN by default

        self.boat_state.__setitem__("current_ui", self.selected_ui)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        #self.screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    
    def shutdown(self):
        pygame.quit()

    def render(self):
        if self.selected_ui == UI_VISION:
            self.ui_vision.render(self.screen)
        elif self.selected_ui == UI_PP:
            self.ui_pp.render(self.screen)

        pygame.display.flip()

    def update(self):
        self.selected_ui = self.boat_state.__getitem__("current_ui")

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.boat_state.set_running(False)
                pygame.quit()
                
        # Ici, utiliser des [if] pour ajouter une nouvelle interface
        if self.selected_ui == UI_VISION:
            self.ui_vision.update()
        elif self.selected_ui == UI_PP:
            self.ui_pp.update()

    def step(self):
        self.update()
        self.render()