from pymavlink import mavutil
import time

# ============================
# PARAMÈTRES
# ============================
CONNECTION_STRING = 'udp:0.0.0.0:14550'
MOTOR_POWER = 10      # % de puissance
TEST_DURATION = 60     # secondes
MOTORS = [3, 4]  # adapte si nécessaire

# ============================
# CONNEXION
# ============================
print("Connexion à BlueOS...")
master = mavutil.mavlink_connection(CONNECTION_STRING)
hb = master.wait_heartbeat(timeout=5)
if not hb:
    raise RuntimeError("Pas de heartbeat sur \f{CONNECTION_STRING}")
print("Heartbeat OK")

print(f"Connecté au système {master.target_system}")

# ============================
# ARMEMENT
# ============================
print("Armement des moteurs...")
master.arducopter_arm()
master.motors_armed_wait()
print("Moteurs armés")

# ============================
# TEST MOTEURS
# ============================
for motor in MOTORS:
    print(f"Test moteur {motor}")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        motor,  # Numéro moteur
        mavutil.mavlink.MOTOR_TEST_THROTTLE_PERCENT,
        MOTOR_POWER,
        TEST_DURATION,
        0, 0, 0
    )

    time.sleep(TEST_DURATION + 1)

# ============================
# DÉSARMEMENT
# ============================
print("Désarmement...")
master.arducopter_disarm()
master.motors_disarmed_wait()
print("Test terminé")
'''
Commandes linux pour setup le réseau ethernet
sudo ip addr flush dev eno1
sudo ip addr add 192.168.2.1/24 dev eno1
sudo ip link set eno1 up
'''
