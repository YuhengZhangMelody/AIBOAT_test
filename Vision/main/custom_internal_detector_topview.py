########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
Custom Object Detection sample (ZED SDK) - MODIFIED
==================================================
This version displays ONLY a 2D top-down (bird's-eye) map of detected objects,
and overlays each object's distance next to its 2D point.

Hotkeys:
  * Zoom in  : 'i'
  * Zoom out : 'o'
  * Quit     : 'q'
"""

import argparse
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pyzed.sl as sl


@dataclass
class MapConfig:
    width: int = 800
    height: int = 800
    pixels_per_meter: float = 20.0  # zoom (px / m)
    max_range_m: float = 50.0        # used for grid / clipping
    show_grid: bool = True


class TopViewMap:
    """
    Simple top-down renderer.
    Coordinate convention (based on RIGHT_HANDED_Y_UP):
      - X: right
      - Y: up
      - Z: forward (away from camera)
    We render a top-down view of the X-Z plane:
      - +X -> right in image
      - +Z -> up in image (towards top)
      - camera is drawn at the bottom center.
    """

    def __init__(self, cfg: MapConfig):
        self.cfg = cfg

    def set_zoom(self, pixels_per_meter: float) -> None:
        self.cfg.pixels_per_meter = float(np.clip(pixels_per_meter, 2.0, 200.0))

    def zoom_in(self) -> None:
        self.set_zoom(self.cfg.pixels_per_meter * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.cfg.pixels_per_meter / 1.25)

    def _world_to_px(self, x_m: float, z_m: float) -> tuple[int, int]:
        # Camera at bottom-center
        cx = self.cfg.width // 2
        cy = self.cfg.height - 40
        px = int(round(cx + x_m * self.cfg.pixels_per_meter))
        py = int(round(cy - z_m * self.cfg.pixels_per_meter))
        return px, py

    def _draw_grid(self, img: np.ndarray) -> None:
        if not self.cfg.show_grid:
            return

        # light grid every 1m, bold every 5m
        meters_per_major = 5
        meters_per_minor = 1

        for m in range(0, int(self.cfg.max_range_m) + 1, meters_per_minor):
            thickness = 1 if (m % meters_per_major) != 0 else 2

            # horizontal (Z)
            x0, y0 = self._world_to_px(-self.cfg.max_range_m, m)
            x1, y1 = self._world_to_px(self.cfg.max_range_m, m)
            cv2.line(img, (x0, y0), (x1, y1), (220, 220, 220, 255), thickness)

            # vertical (X) - symmetric around 0
            x_left, y_top = self._world_to_px(-m, self.cfg.max_range_m)
            x_left2, y_bot = self._world_to_px(-m, 0)
            cv2.line(img, (x_left, y_top), (x_left2, y_bot), (220, 220, 220, 255), thickness)

            x_right, y_top = self._world_to_px(m, self.cfg.max_range_m)
            x_right2, y_bot = self._world_to_px(m, 0)
            cv2.line(img, (x_right, y_top), (x_right2, y_bot), (220, 220, 220, 255), thickness)

        # Axes
        # Z axis (forward)
        x0, y0 = self._world_to_px(0, 0)
        x1, y1 = self._world_to_px(0, self.cfg.max_range_m)
        cv2.arrowedLine(img, (x0, y0), (x1, y1), (80, 80, 80, 255), 2, tipLength=0.02)
        # X axis (right)
        x2, y2 = self._world_to_px(self.cfg.max_range_m, 0)
        cv2.arrowedLine(img, (x0, y0), (x2, y2), (80, 80, 80, 255), 2, tipLength=0.02)

        cv2.putText(img, "Z (forward)", (x1 + 6, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "X (right)", (x2 - 80, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50, 255), 1, cv2.LINE_AA)

    def render(self, objects: sl.Objects) -> np.ndarray:
        img = np.full((self.cfg.height, self.cfg.width, 4), (245, 239, 239, 255), dtype=np.uint8)

        # grid + camera marker
        self._draw_grid(img)

        cam_px = (self.cfg.width // 2, self.cfg.height - 40)
        cv2.circle(img, cam_px, 6, (30, 30, 30, 255), -1)
        cv2.putText(img, "CAM", (cam_px[0] + 10, cam_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30, 255), 1, cv2.LINE_AA)

        # objects
        # objects.object_list: list[sl.ObjectData]
        for obj in getattr(objects, "object_list", []):
            # Skip if not tracked (optional)
            # tracking_state can be OK, SEARCHING, OFF
            try:
                if hasattr(obj, "tracking_state") and obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                    continue
            except Exception:
                pass

            pos = obj.position  # sl.float3
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

            # Distance: ZED states distance is metric from left camera to object position (euclidean)
            dist = math.sqrt(x * x + y * y + z * z)

            # Clip to max range to avoid drawing off-screen
            if z < 0:
                # Behind camera
                continue
            if dist > (self.cfg.max_range_m * 1.2):
                continue

            px, py = self._world_to_px(x, z)
            if px < 0 or px >= self.cfg.width or py < 0 or py >= self.cfg.height:
                continue

            # marker
            cv2.circle(img, (px, py), 5, (0, 120, 255, 255), -1)

            # label that "follows" the point (drawn next to it)
            obj_id = getattr(obj, "id", -1)
            label = f"{dist:.1f} m"
            if obj_id is not None and obj_id >= 0:
                label = f"#{obj_id}  {label}"

            # Place text with a small offset so it follows the point without covering it
            tx, ty = px + 8, py - 8
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20, 255), 2, cv2.LINE_AA)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1, cv2.LINE_AA)

        # HUD
        cv2.putText(
            img,
            f"Zoom: {self.cfg.pixels_per_meter:.1f} px/m",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30, 255),
            2,
            cv2.LINE_AA,
        )
        return img


def run(opt: argparse.Namespace) -> None:
    print("Initializing Camera...")
    zed = sl.Camera()

    # --- Stream via IP ---
    ip_str = (opt.ip_address or "").strip()
    if not ip_str:
        raise SystemExit("Error: --ip_address requis (format a.b.c.d:port ou a.b.c.d).")

    parts = ip_str.split(":")
    input_type = sl.InputType()
    if len(parts) == 2:
        input_type.set_from_stream(parts[0], int(parts[1]))
        print("[Sample] Using Stream input, IP:PORT :", ip_str)
    else:
        input_type.set_from_stream(parts[0])
        print("[Sample] Using Stream input, IP :", ip_str)

    init_params = sl.InitParameters(input_t=input_type)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    # --- OpenCV calibration file ---
    calib = (opt.calib_yml or "").strip()
    if not calib:
        raise SystemExit("Error: --calib_yml requis")
    if not os.path.exists(calib):
        raise SystemExit(f"Error: calibration file not found: {calib}")
    init_params.optional_opencv_calibration_file = calib
    print("[Calib] Using OpenCV calibration file:", calib)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Camera Open : {repr(status)}. Exit program.")
    print("Initializing Camera... DONE")

    # Enable positional tracking module (recommended for stable 3D / tracking)
    print("Enabling Positional Tracking...")
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    status = zed.enable_positional_tracking(positional_tracking_parameters)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise SystemExit(f"Positional Tracking enable : {repr(status)}. Exit program.")
    print("Enabling Positional Tracking... DONE")

    # Enable object detection module
    print("Enabling Object Detection...")
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    obj_param.custom_onnx_file = opt.custom_onnx
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    status = zed.enable_object_detection(obj_param)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise SystemExit(f"Object Detection enable : {repr(status)}. Exit program.")
    print("Enabling Object Detection... DONE")

    # Custom OD runtime parameters
    detection_parameters_rt = sl.CustomObjectDetectionRuntimeParameters()
    detection_parameters_rt.object_detection_properties.detection_confidence_threshold = 30

    # Example: override per-class properties (keep your original sample structure)
    props_dict = {1: sl.CustomObjectDetectionProperties(), 2: sl.CustomObjectDetectionProperties()}
    props_dict[1].native_mapped_class = sl.OBJECT_SUBCLASS.PERSON
    props_dict[1].object_acceleration_preset = sl.OBJECT_ACCELERATION_PRESET.MEDIUM
    props_dict[1].detection_confidence_threshold = 40
    props_dict[2].detection_confidence_threshold = 50
    props_dict[2].max_allowed_acceleration = 10 * 10
    detection_parameters_rt.object_class_detection_properties = props_dict

    # Grab runtime parameters:
    # IMPORTANT: object 3D positions depend on this reference frame. We want camera-relative top view.
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA  # see ZED docs

    objects = sl.Objects()

    # 2D map UI
    map_cfg = MapConfig(max_range_m=float(init_params.depth_maximum_distance))
    top_view = TopViewMap(map_cfg)
    window_name = "ZED | 2D Top View (Objects)"

    if not opt.disable_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nHotkeys: i=zoom in, o=zoom out, q=quit\n")

    quit_bool = False
    while not quit_bool:
        if zed.grab(runtime_parameters) > sl.ERROR_CODE.SUCCESS:
            break

        status = zed.retrieve_custom_objects(objects, detection_parameters_rt)
        if status != sl.ERROR_CODE.SUCCESS:
            continue

        if opt.disable_gui:
            continue

        map_img = top_view.render(objects)
        cv2.imshow(window_name, map_img)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            quit_bool = True
        elif key == ord('i'):
            top_view.zoom_in()
        elif key == ord('o'):
            top_view.zoom_out()

    if not opt.disable_gui:
        cv2.destroyAllWindows()

    zed.disable_object_detection()
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--custom_onnx', type=str, required=True, help='Path to custom ONNX model to use')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default='')
    parser.add_argument('--calib_yml', type=str, required=True, help='Chemin vers le fichier calibration OpenCV .yml')

    parser.add_argument('--disable_gui', action='store_true', help='Disable OpenCV GUI to increase detection performance')
    opt = parser.parse_args()

    run(opt)
