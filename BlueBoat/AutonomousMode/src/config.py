from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

black_ip = "127.0.0.1"
black_port = 34010
grey_ip = "127.0.0.1"
grey_port = 34000

grey_id_cam = 0
black_id_cam = 1

onnx = PROJECT_ROOT / "model" / "yolo11n.onnx"

grey_calib  = PROJECT_ROOT / "calibrations" / "zed_calibration_grey.yml"
black_calib = PROJECT_ROOT / "calibrations" / "zed_calibration_black_171225.yml"
