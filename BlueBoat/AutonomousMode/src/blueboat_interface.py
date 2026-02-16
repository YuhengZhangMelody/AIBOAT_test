# blueboat_mavlink.py
import time
import math
from threading import Thread
from pymavlink import mavutil
from boat_state import ThreadSafeBoatState

MOTOR_POWER = 10      # % de puissance
TEST_DURATION = 10     # secondes
MOTORS = [3, 4]  # adapte si nécessaire

class BlueBoatMavlink(Thread):
    """
    Thread de communication MAVLink avec le BlueBoat.
    Lit les messages MAVLink et met à jour le boat_state.
    """

    LOOP_SPEED_HZ = 10  # fréquence de lecture MAVLink (Hz)

    def __init__(
        self,
        boat_state: ThreadSafeBoatState,
        connection_string: str = "udp:0.0.0.0:14550",
    ):
        
        self.boat_state = boat_state
        self.connection_string = connection_string
        self.master = None
        
        # ---------- connexion ----------
        print("[MAVLINK] Connecting to BlueBoat...")
        self.master = mavutil.mavlink_connection(self.connection_string)
        self.master.wait_heartbeat()
        print(
            f"[MAVLINK] Connected to system {self.master.target_system}, "
            f"component {self.master.target_component}"
        )
        # ============================
        # ARMEMENT
        # ============================
        print("Armement des moteurs...")
        self.master.arducopter_arm()
        self.master.motors_armed_wait()
        print("Moteurs armés")

    
    # ---------- lecture messages ----------
    def _read_messages(self):
        """
        Lit tous les messages disponibles sans bloquer
        """
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            self._handle_message(msg)

    def _handle_message(self, msg):
        mtype = msg.get_type()

        # -------- IMU --------
        if mtype == "RAW_IMU":
            self.boat_state["accel"] = (msg.xacc, msg.yacc, msg.zacc)
            self.boat_state["gyro"] = (msg.xgyro, msg.ygyro, msg.zgyro)
            self.boat_state["mag"] = (msg.xmag, msg.ymag, msg.zmag)

        # -------- Attitude --------
        elif mtype == "ATTITUDE":
            self.boat_state["attitude"]["roll"] = msg.roll  # radians
            self.boat_state["attitude"]["pitch"] = msg.pitch  # radians
            self.boat_state["attitude"]["yaw"] = msg.yaw  # radians

            self.boat_state["heading"] = msg.yaw  # radians

        # -------- GPS --------
        elif mtype == "GPS_RAW_INT":
            self.boat_state["latitude"] = msg.lat / 1e7
            self.boat_state["longitude"] = msg.lon / 1e7
            self.boat_state["altitude"] = msg.alt / 1000.0

        # -------- Battery --------
        elif mtype == "BATTERY_STATUS":
            if msg.energy_consumed != -1:
                self.boat_state["consumed_charge"] = msg.energy_consumed

    # ---------- boucle thread ----------

    def _send_command(self):
        for motor in MOTORS:
            print(f"Test moteur {motor}")

            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
                0,
                motor,  # Numéro moteur
                mavutil.mavlink.MOTOR_TEST_THROTTLE_PERCENT,
                MOTOR_POWER,
                TEST_DURATION,
                0, 0, 0
            )

            time.sleep(TEST_DURATION + 1)

    def step(self):
        
        self._read_messages()
        self._send_command()
        
