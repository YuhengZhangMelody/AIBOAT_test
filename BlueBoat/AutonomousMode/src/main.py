from boat_state import ThreadSafeBoatState
from interface import UISystem
from blueboat_interface import BlueBoatMavlink
from vision import Vision
from path_planning import PathPlanning

import subprocess
import os

from time import time, sleep
from threading import Thread

def vision_setup():
    #Attribution de l'addresse ip pour communiquer avec le blue boat et la station à terre
    os.system("sudo ip addr flush dev eno1")
    os.system("sudo ip addr add 192.168.2.1 dev eno1")
    os.system("sudo ip link set eno1 up")
    
    #Streaming des caméras
    print("stop zed_media_server_cli.service...")
    os.system('sudo systemctl stop zed_media_server_cli.service')
    print("restart zed_x_daemon.service...")
    os.system('sudo systemctl restart zed_x_daemon.service')
    
    print("Open ZED_Media_server and stream cameras...")
    PASSWORD = "tameo\n"
    WAITING_FOR = "[Streaming] Streaming is now running...."

    process_cam = subprocess.Popen(["sudo", "-S", "ZED_Media_Server", "--cli"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,)
    # Send password once
    process_cam.stdin.write(PASSWORD)
    process_cam.stdin.flush()
    c=0
    for line in process_cam.stdout:
        if WAITING_FOR in line and c==0:
            print("first camera opened successfully")
            c+=1
        if WAITING_FOR in line and c==1:
            print("second camera opened successfully")
            break
    sleep(2)

def run_at(subsys, target_hz, boat_state: ThreadSafeBoatState):
    while boat_state.running():
        begin = time()
        subsys.step()
        duration = time() - begin
        if duration < 1/float(target_hz):
            sleep(1/float(target_hz) - duration)
    
if __name__ == "__main__":

    
    boat_state = ThreadSafeBoatState()
    vision_setup()

    # Initialisation des sous-systèmes
    ui_system = UISystem(boat_state=boat_state)
    BlueBoatMavlink_system = BlueBoatMavlink(boat_state=boat_state,connection_string="udp:0.0.0.0:14550")
    vision_system = Vision(boat_state=boat_state)
    pathPlaning_system = PathPlanning(boat_state=boat_state)
    subsystems = [BlueBoatMavlink_system,vision_system,pathPlaning_system] #, canbus_system, record_data_system
    #subsystems = [pathPlaning]

    # Création des threads secondaires
    threads = []
    for sys in subsystems:
        threads.append(Thread(
            target=lambda s=sys: run_at(s, s.LOOP_SPEED_HZ, boat_state)
        ))
    
    #Lancement du thread principal
    for t in threads:
        t.start()
    
    # Le sous-système "ui" est spécial car les appels à pygames
    # doivent êtres effectués sur le thread principal
    run_at(ui_system, ui_system.LOOP_SPEED_HZ, boat_state)

    # Arrêt
    for t in threads:
        t.join()

    for sys in subsystems:
        sys.shutdown()

    ui_system.shutdown()
        
        
        
        