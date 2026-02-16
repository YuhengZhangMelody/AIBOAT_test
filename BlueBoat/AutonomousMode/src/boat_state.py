import sys
from pathlib import Path

#Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.config as config

from threading import RLock

from src.path_planning_utils.classes import Boat, Buoy, Polygon, Waypoint
from src.path_planning_utils.ekf_slam import Slam
import numpy as np

class ThreadSafeBoatState:
    def __init__(self):
        # Protège l'accès aux données depuis plusieurs threads
        self.data_lock = RLock()
        #self.data_lock = Lock()
        self._running = True
        
        # Données en elles-mêmes
        self._init_data()

    def _init_data(self):
        self._data = {
            
            # Latitude, longitude, altitude
            "latitude":48.831475459292776,
            "longitude":2.225964454415254,
            "altitude" : 0,
        
            # IMU
            "accel": [0.0, 0.0, 0.0],
            "gyro":  [0.0, 0.0, 0.0],
            "mag":   [0.0, 0.0, 0.0],

            # Angles
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},  # radians
            "heading": 0.0,  # radians, 0 = +x, CCW positif
            

            #Vision
            "detected_objects": [], # liste de (x, y, z, dist) en repère capteur
            "detected_objects_boat":[], #Dans le référentiel du bateau 
            #[{"ObjId":cid, "IdCam":config.black_id_cam, 'x':-z, 'y':x, 'z':y, 'dist':dist}]
            # y -> devant, z en haut, x à droite

            "Cameras": {
                config.grey_id_cam : {"XCam": 0.0, "YCam": 0.36, "YawCam_deg": 0.0, "FovCam_deg": 60.0},
                config.black_id_cam: {"XCam": 0.0, "YCam": -0.36, "YawCam_deg": 0.0, "FovCam_deg": 60.0},
            }, #Xcam et Ycam sont dans le repère bateau

            # angle du volant, int entre 0 et 1023
            "direction": 0,

            # manette des gaz, int entre 0 et 1023
            "duty": 0.0,

            # informations BMS
            # TODO: Voirs quelles informations peuvent êtres récoltées
            "temp": [0,0], # ne peut pas encore être récolté mais on en a besoin pour l'ui
            "power": 0.0, # idem
            "power_max": 100.0, # idem
            "voltage":0.0,
            "batt_voltage" : 52.5,
            "SOC" : 95.0,
            "max_charge": 10000.0, # en Wh
            "consumed_charge": 0.0, # en Wh
            "total_input_current":0.0, #Courant total à l'entrée des moteurs
            "current" : 0.0,

            # VESC moteur arrière (motor back = mtr_b)
            "mtr_b_duty": 0,
            "mtr_b_input_current": 0,
            "mtr_b_motor_current": 0,
            "mtr_b_input_voltage": 0,
            "mtr_b_temp": 0,
            "mtr_b_pelec" : 0,
            "mtr_b_pmeca" : 0,
            "mtr_b_yield": 0.0,
            "mtr_b_erpm":0.0,
            
            # VESC moteur avant droite (motor front right = mtr_fr)
            "mtr_fr_duty": 0,
            "mtr_fr_input_current": 0,
            "mtr_fr_motor_current": 0,
            "mtr_fr_input_voltage": 0,
            "mtr_fr_temp": 0,
            "mtr_fr_pelec" : 0,
            "mtr_fr_pmeca" : 0,
            "mtr_fr_yield": 0.0,
            "mtr_fr_erpm":0.0,

            # VESC moteur avant left (motor front left = mtr_fl)
            "mtr_fl_duty": 0,
            "mtr_fl_input_current": 0,
            "mtr_fl_motor_current": 0,
            "mtr_fl_input_voltage": 0,
            "mtr_fl_temp": 0,
            "mtr_fl_pelec" : 0,
            "mtr_fl_pmeca" : 0,
            "mtr_fl_yield": 0.0,
            "mtr_fl_erpm":0.0,

            #Caractéristiques moteurs
            "mtr_b_nb_poles": 5,
            "mtr_fr_nb_poles": 3,
            "mtr_fl_nb_poles": 3,

            "mtr_b_kt": 0.042465,
            "mtr_fr_kt": 0.07218,
            "mtr_fl_kt": 0.07218,

            "total_motor_pelec": 0.0,
            "total_motor_pmeca": 0.0,
            "global_motor_yield": 0.0,

            # boutons (tous des int, 0=désactivé, 1=activé)

            # activation du différentiel
            "diff": 0,
            "current_ui": 0, # 0=debug, 1=...
            "nb_ui": 3,
            # TODO: finir de les lister

            "buttons":[0,0,0,0,0,0,0], #[changement ui,ND,ND,ND,ND,diff1,diff2]
            "runLog": False,
            "wasLog": False,

            #Moyenne glissante de la vitesse
            "speed_avg": 0.0,
            "lats":[],
            "lons":[],

            ##--------------------Path Planning (PP)-------------------##
            "boatObject_pp": Boat(0,0,0), # objet Boat de path_planning_utils/classes.py
            "bool_final_goal_reached_pp": False, # simple bool pour savoir si le goal final a été atteint ou pas
            "numpyArray_predicted_trajectory_pp": np.array([[0,0,0,0,0], [0,0,0,0,0]]), # trajectoire prédite par le DWA pour les prochaines secondes
            "waypointObjectList_pp": [], # liste de waypoints à atteindre
            "buoyObjectList_pp": [],# liste de bouées présentes sur la carte
            "time_pp": 0.0,
            "numpyArray_estimated_trajectory_pp": np.array([]), # trajectoire estimée par le SLAM EKF
            "numpyArray_true_trajectory_pp": np.array([]), # temporaire
            "slamObject_pp": Slam(np.array([]), np.array([]), Boat(0,0,0)),
        }

    def running(self):
        with self.data_lock:
            return self._running
        
    def set_running(self, value: bool):
        with self.data_lock:
            self._running = value

    def __setitem__(self, name, value):
        with self.data_lock:
            self._data[name] = value

    def __getitem__(self, name):
        with self.data_lock:
            return self._data.get(name, None)