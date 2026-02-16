import math 

def _cam_to_boat(x,y,Xcam_boat, Ycam_boat, YawCam_rad):
    # Convert camera position from boat frame to world frame
    X_boat = (x * math.cos(YawCam_rad) - y * math.sin(YawCam_rad))+Xcam_boat
    Y_boat = (x * math.sin(YawCam_rad) + y * math.cos(YawCam_rad))+Ycam_boat
    return X_boat, Y_boat


def _zed_to_cam(x_zed,y_zed,z_zed):
    return x_zed, -z_zed, -y_zed