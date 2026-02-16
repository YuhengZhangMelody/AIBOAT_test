#!/usr/bin/env python3
import sys
import subprocess
import os 

from pathlib import Path

#Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from threading import Thread
import time
import math
import pyzed.sl as sl
import math
from boat_state import ThreadSafeBoatState
import src.config as config

from src.vision_utils.vision_utils import _zed_to_cam,_cam_to_boat

import time

# ====== Variables modifiables dans le code (pas d'arguments CLI) ======
CONF_THRESHOLD = 80      # [0..100]
PRINT_HZ = 60.0          # fréquence des prints
INPUT_SIZE = 512         # taille d'entrée du modèle ONNX (ex: 512 pour 1x3x512x512)
# =====================================================================

# COCO 80 classes (Ultralytics order)
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


class Vision(Thread):

    LOOP_SPEED_HZ = 60

    def coco_name(class_id: int) -> str:
        return COCO80[class_id] if 0 <= class_id < len(COCO80) else f"unknown({class_id})"

    def __init__(
        self,
        boat_state: ThreadSafeBoatState,
    ):
        self.boat_state = boat_state

        self.XBlackCam_boat = self.boat_state["Cameras"][config.black_id_cam]["XCam_boat"]
        self.YBlackCam_boat = self.boat_state["Cameras"][config.black_id_cam]["YCam_boat"]
        self.YawBlackCam_deg = self.boat_state["Cameras"][config.black_id_cam]["YawCam_deg"]
    
        self.XGreyCam_boat = self.boat_state["Cameras"][config.grey_id_cam]["XCam_boat"]
        self.YGreyCam_boat = self.boat_state["Cameras"][config.grey_id_cam]["YCam_boat"]
        self.YawGreyCam_deg = self.boat_state["Cameras"][config.grey_id_cam]["YawCam_deg"]

        #Ouverture des caméras et configurations
        try:
            self.zed_black, self.rt_black, self.objects_black = self.open_zed_stream(
                "ZED_BLACK", config.black_ip, config.black_port, str(config.onnx), str(config.black_calib)
            )
            self.zed_grey, self.rt_grey, self.objects_grey = self.open_zed_stream(
                "ZED_GREY", config.grey_ip, config.grey_port, str(config.onnx), str(config.grey_calib)
            )
        except Exception as e:
            
            raise RuntimeError(f"Erreur : {e}")

        print("[INFO] 2x Stream + ONNX detection OK. CTRL+C pour arrêter.")
        print(f"[INFO] BLACK: {config.black_ip}:{config.black_port}")
        print(f"[INFO] GREY : {config.grey_ip}:{config.grey_port}")

        

    
    def open_zed_stream(self, name: str, ip: str, port: int, onnx_path: str, calib_path: str = ""):
        """
        Ouvre une ZED en stream (IP:PORT), active tracking + custom object detection.
        Retourne: (zed, runtime_params, objects_container)
        """
        zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_stream(ip, port)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_maximum_distance = 50

        if calib_path:
            init_params.optional_opencv_calibration_file = calib_path

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"[{name}] zed.open({ip}:{port}) -> {err}")

        # POSITIONAL TRACKING
        pt = sl.PositionalTrackingParameters()
        pt.set_as_static = True
        err = zed.enable_positional_tracking(pt)
        if err != sl.ERROR_CODE.SUCCESS:
            zed.close()
            raise RuntimeError(f"[{name}] enable_positional_tracking() -> {err}")

        # OBJECT DETECTION (CUSTOM YOLOLIKE)
        od_params = sl.ObjectDetectionParameters()
        od_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
        od_params.custom_onnx_file = onnx_path
        od_params.max_range = 30
        od_params.enable_tracking = True
        od_params.enable_segmentation = True

        err = zed.enable_object_detection(od_params)
        if err != sl.ERROR_CODE.SUCCESS:
            zed.disable_positional_tracking()
            zed.close()
            raise RuntimeError(f"[{name}] enable_object_detection() -> {err}")

        # RUNTIME PARAMS (CUSTOM DETECTOR)
        rt = sl.CustomObjectDetectionRuntimeParameters()
        rt.object_detection_properties.detection_confidence_threshold = CONF_THRESHOLD

        objects = sl.Objects()
        return zed, rt, objects

    
    def step(self):
        

        # ---- GRAB + RETRIEVE (indépendant) ----
        got_black = (self.zed_black.grab() == sl.ERROR_CODE.SUCCESS)
        if got_black:
            self.zed_black.retrieve_custom_objects(self.objects_black, self.rt_black)

        got_grey = (self.zed_grey.grab() == sl.ERROR_CODE.SUCCESS)
        if got_grey:
            self.zed_grey.retrieve_custom_objects(self.objects_grey, self.rt_grey)

        # ---- MAP DATA ----
        # Note: show_map doit accepter des points avec couleur: (x, z, dist, "black"/"grey")
        Obj_detected = []
        Obj_detected_boat = []

        if got_black and len(self.objects_black.object_list) > 0:
            for o in self.objects_black.object_list:
                cid = int(o.raw_label)
                x_cam, y_cam, z_cam = _zed_to_cam(float(o.position[0]), float(o.position[1]), float(o.position[2]))
                
                dist = math.sqrt(x_cam*x_cam + y_cam*y_cam)
                Obj_detected.append({"ObjId":cid, "IdCam":config.black_id_cam, 'x':x_cam, 'y':y_cam, 'z':z_cam, 'dist':dist})
                x_boat,y_boat = _cam_to_boat(x_cam,y_cam,self.XBlackCam_boat,self.YBlackCam_boat,self.YawBlackCam_deg)
                Obj_detected_boat.append({"ObjId":cid, "IdCam":config.black_id_cam, 'x':x_boat, 'y':y_boat, 'z':z_cam, 'dist':dist})

        if got_grey and len(self.objects_grey.object_list) > 0:
            for o in self.objects_grey.object_list:
                cid = int(o.raw_label)
                x_cam, y_cam, z_cam = _zed_to_cam(float(o.position[0]), float(o.position[1]), float(o.position[2]))
                dist = math.sqrt(x_cam*x_cam + y_cam*y_cam)
                Obj_detected.append({"ObjId":cid, "IdCam":config.grey_id_cam, 'x':x_cam, 'y':y_cam, 'z':z_cam, 'dist':dist})
                x_boat,y_boat = _cam_to_boat(x_cam,y_cam,self.XGreyCam_boat,self.YGreyCam_boat,self.YawGreyCam_deg)
                Obj_detected_boat.append({"ObjId":cid, "IdCam":config.grey_id_cam, 'x':x_boat, 'y':y_boat, 'z':z_cam, 'dist':dist})

        #self.boat_state["detected_objects"] = Obj_detected
        self.boat_state["detected_objects_boat"] = Obj_detected_boat

        #self.boat_state["detected_objects_boat"] = Obj_detected
def safe_close(self, name: str, zed: sl.Camera):
    try:
        zed.disable_object_detection()
    except Exception:
        pass
    try:
        zed.disable_positional_tracking()
    except Exception:
        pass
    try:
        zed.close()
    except Exception:
        pass
    print(f"[INFO] {name} closed.")

def shutdown(self):
    self.safe_close("ZED_BLACK", self.zed_black)
    self.safe_close("ZED_GREY" , self.zed_grey )



