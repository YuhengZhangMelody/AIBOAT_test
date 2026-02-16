#!/usr/bin/env python3
import argparse
import time
import math
import pyzed.sl as sl
import math

from map import show_map

# ====== Variables modifiables dans le code (pas d'arguments CLI) ======
CONF_THRESHOLD = 80      # [0..100]
PRINT_HZ = 3.0          # fréquence des prints
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

def coco_name(class_id: int) -> str:
    return COCO80[class_id] if 0 <= class_id < len(COCO80) else f"unknown({class_id})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="IP du sender (stream ZED)")
    parser.add_argument("--port", type=int, required=True, help="Port du sender (stream ZED)")
    parser.add_argument("--onnx", required=True, help="Chemin vers le modèle ONNX (custom YOLO-like)")
    parser.add_argument("--calib", default="", help="Chemin vers le fichier de calibration OpenCV (xml/yaml/json). Optionnel.")
    args = parser.parse_args()

    zed = sl.Camera()

    init_params = sl.InitParameters()
    # Stream input (IP + PORT) :contentReference[oaicite:2]{index=2}
    init_params.set_from_stream(args.ip, args.port)
    
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 50
    
    # Calibration OpenCV optionnelle :contentReference[oaicite:3]{index=3}
    if args.calib:
        init_params.optional_opencv_calibration_file = args.calib

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] zed.open() -> {err}")
        return 1

    ###
    #POSITIONNAL_TRACKING

    positional_tracking_params = sl.PositionalTrackingParameters()
    positional_tracking_params.set_as_static = True
    status = zed.enable_positional_tracking(positional_tracking_params)

    ###
    #OBJECT_DETECTION_PARAMETERS

    od_params = sl.ObjectDetectionParameters()
    
    #Model
    od_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    od_params.custom_onnx_file = args.onnx

    #Paramètres
    od_params.max_range = 30

    od_params.enable_tracking = True
    od_params.enable_segmentation = True 
    

    err = zed.enable_object_detection(od_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] enable_object_detection() -> {err}")
        zed.close()
        return 2

    #RUNTIME_PARAMETERS pour les modèles de détection non zed (ici c'est un yolo)
    detection_parameters_rt = sl.CustomObjectDetectionRuntimeParameters()
    # Default properties, apply to all object class
    detection_parameters_rt.object_detection_properties.detection_confidence_threshold = CONF_THRESHOLD
    
    objects = sl.Objects()

    period = 1.0 / max(PRINT_HZ, 0.1)
    last_print = 0.0

    print("[INFO] Stream + ONNX detection OK. CTRL+C pour arrêter.")
    print(f"[INFO] Stream: {args.ip}:{args.port}")
    if args.calib:
        print(f"[INFO] OpenCV calib: {args.calib}")
    print(f"[INFO] ONNX: {args.onnx} | INPUT_SIZE={INPUT_SIZE} | CONF_THRESHOLD={CONF_THRESHOLD}")
    
    try:
        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            status = zed.retrieve_custom_objects(objects, detection_parameters_rt)

            now = time.time()
            if now - last_print < period:
                continue
            last_print = now

            Pos = []

            if  len(objects.object_list) > 0:
                print(f"\n[DETECTED] {len(objects.object_list)}")
                for o in objects.object_list:
                    
                    x, y, z = float(o.position[0]), float(o.position[1]), float(o.position[2])
                    dist = math.sqrt(x*x + y*y + z*z)
                    Pos.append((x,z,dist))

                    cid = int(o.raw_label)  # IMPORTANT: raw_label pour custom detector
                    
                    print(f"id={o.id} conf={o.confidence} class_name={coco_name(cid)} pos={tuple(map(float, o.position))} dist={dist}")

            else:
                print("[DETECTED] 0")

            show_map(Pos,math.pi/2)

    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé.")

    zed.disable_object_detection()
    zed.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
