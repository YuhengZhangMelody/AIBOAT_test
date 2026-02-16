#!/usr/bin/env python3
import argparse
import time
import math
import pyzed.sl as sl

from map import show_map

# ====== Variables modifiables dans le code (pas d'arguments CLI) ======
CONF_THRESHOLD = 80      # [0..100]
PRINT_HZ = 3.0           # fréquence des prints
INPUT_SIZE = 512         # taille d'entrée du modèle ONNX (ex: 512 pour 1x3x512x512)
HEADING_RAD = math.pi/2  # direction du cône FOV sur la map (radians)
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


def open_zed_stream(name: str, ip: str, port: int, onnx_path: str, calib_path: str = ""):
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


def safe_close(name: str, zed: sl.Camera):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--black_ip", required=True, help="IP du sender stream ZED BLACK")
    parser.add_argument("--black_port", type=int, required=True, help="Port du sender stream ZED BLACK")
    parser.add_argument("--grey_ip", required=True, help="IP du sender stream ZED GREY")
    parser.add_argument("--grey_port", type=int, required=True, help="Port du sender stream ZED GREY")
    parser.add_argument("--black_calib", default="", help="Chemin vers le fichier de calibration OpenCV. Optionnel.")
    parser.add_argument("--grey_calib", default="", help="Chemin vers le fichier de calibration OpenCV. Optionnel.")

    parser.add_argument("--onnx", required=True, help="Chemin vers le modèle ONNX (custom YOLO-like)")
    
    args = parser.parse_args()

    period = 1.0 / max(PRINT_HZ, 0.1)
    last_print = 0.0

    try:
        zed_black, rt_black, objects_black = open_zed_stream(
            "ZED_BLACK", args.black_ip, args.black_port, args.onnx, args.black_calib
        )
        zed_grey, rt_grey, objects_grey = open_zed_stream(
            "ZED_GREY", args.grey_ip, args.grey_port, args.onnx, args.grey_calib
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print("[INFO] 2x Stream + ONNX detection OK. CTRL+C pour arrêter.")
    print(f"[INFO] BLACK: {args.black_ip}:{args.black_port}")
    print(f"[INFO] GREY : {args.grey_ip}:{args.grey_port}")

    
    print(f"[INFO] ONNX: {args.onnx} | INPUT_SIZE={INPUT_SIZE} | CONF_THRESHOLD={CONF_THRESHOLD}")

    try:
        while True:

            # ---- GRAB + RETRIEVE (indépendant) ----
            got_black = (zed_black.grab() == sl.ERROR_CODE.SUCCESS)
            if got_black:
                zed_black.retrieve_custom_objects(objects_black, rt_black)

            got_grey = (zed_grey.grab() == sl.ERROR_CODE.SUCCESS)
            if got_grey:
                zed_grey.retrieve_custom_objects(objects_grey, rt_grey)

            # ---- MAP DATA ----
            # Note: show_map doit accepter des points avec couleur: (x, z, dist, "black"/"grey")
            Pos = []

            if got_black and len(objects_black.object_list) > 0:
                for o in objects_black.object_list:
                    x, y, z = float(o.position[0]), float(o.position[1]), float(o.position[2])
                    dist = math.sqrt(x*x + y*y + z*z)
                    Pos.append((x, z, dist, "black"))

            if got_grey and len(objects_grey.object_list) > 0:
                for o in objects_grey.object_list:
                    x, y, z = float(o.position[0]), float(o.position[1]), float(o.position[2])
                    dist = math.sqrt(x*x + y*y + z*z)
                    Pos.append((x, z, dist, "grey"))

            # ---- PRINT (throttled) ----
            now = time.time()
            if now - last_print >= period:
                last_print = now
                nb_black = len(objects_black.object_list) if got_black else 0
                nb_grey = len(objects_grey.object_list) if got_grey else 0
                print(f"\n[DETECTED] BLACK={nb_black} | GREY={nb_grey}")

                if got_black:
                    for o in objects_black.object_list:
                        cid = int(o.raw_label)
                        x, y, z = float(o.position[0]), float(o.position[1]), float(o.position[2])
                        dist = math.sqrt(x*x + y*y + z*z)
                        print(f"[BLACK] id={o.id} conf={o.confidence} class={coco_name(cid)} pos={tuple(map(float, o.position))} dist={dist:.2f}m")

                if got_grey:
                    for o in objects_grey.object_list:
                        cid = int(o.raw_label)
                        x, y, z = float(o.position[0]), float(o.position[1]), float(o.position[2])
                        dist = math.sqrt(x*x + y*y + z*z)
                        print(f"[GREY ] id={o.id} conf={o.confidence} class={coco_name(cid)} pos={tuple(map(float, o.position))} dist={dist:.2f}m")

            # ---- DISPLAY ----
            #show_map(Pos, HEADING_RAD)

    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé.")
    finally:
        safe_close("ZED_BLACK", zed_black)
        safe_close("ZED_GREY", zed_grey)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
