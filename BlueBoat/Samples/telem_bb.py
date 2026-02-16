from pymavlink import mavutil
import time
''' Code To Fetch all the parameters (PID, Max turn angles, ... /!\ il y en a plus d'une centaine)
# Connect to BlueOS MAVLink over Ethernet (UDP)
# BlueOS default MAVLink endpoint
CONNECTION_STRING = 'udp:0.0.0.0:14550'

print("Connecting to vehicle...")
master = mavutil.mavlink_connection(CONNECTION_STRING)

# Wait for heartbeat (confirms connection)
master.wait_heartbeat()
print(f"Connected to system {master.target_system}, component {master.target_component}")

# Request all parameters
print("Requesting parameters...")
master.mav.param_request_list_send(
    master.target_system,
    master.target_component
)

parameters = {}

# Receive parameters
start_time = time.time()
timeout = 30  # seconds

while True:
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
    if not msg:
        break

    param_id = msg.param_id.strip('\x00')
    parameters[param_id] = msg.param_value

    print(f"{param_id} = {msg.param_value}")

    # Stop when all parameters are received
    if len(parameters) >= msg.param_count:
        break

    # Safety timeout
    if time.time() - start_time > timeout:
        print("Timeout while receiving parameters")
        break

print(f"\nReceived {len(parameters)} parameters.")
'''

# Connect to BlueBoat via Ethernet (UDP)
# Option 1: Listen to broadcast (recommended)
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')

# Option 2: Direct IP connection
# master = mavutil.mavlink_connection('udp:192.168.2.2:14550')

print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected to system:", master.target_system)

def read_imu():
    msg = master.recv_match(type='RAW_IMU', blocking=True, timeout=1)
    if msg:
        return {
            "accel": (msg.xacc, msg.yacc, msg.zacc),  # mg
            "gyro": (msg.xgyro, msg.ygyro, msg.zgyro),  # mrad/s
            "mag": (msg.xmag, msg.ymag, msg.zmag)
        }

def read_attitude():
    msg = master.recv_match(type='ATTITUDE', blocking=True, timeout=1)
    if msg:
        return {
            "roll": msg.roll,
            "pitch": msg.pitch,
            "yaw": msg.yaw
        }

def read_depth():
    msg = master.recv_match(type='SCALED_PRESSURE2', blocking=True, timeout=1)
    if msg:
        return {
            "pressure": msg.press_abs,  # hPa
            "temperature": msg.temperature / 100.0  # °C
        }

def read_gps():
    msg = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=1)
    if msg:
        return {
            "lat": msg.lat / 1e7,
            "lon": msg.lon / 1e7,
            "alt": msg.alt / 1000.0,
            "fix_type": msg.fix_type,
            "satellites": msg.satellites_visible
        }

def read_battery():
    msg = master.recv_match(type='BATTERY_STATUS', blocking=True, timeout=1)
    if not msg:
        return None

    return {
        "voltage_per_cell": [v / 1000.0 for v in msg.voltages if v != 65535],
        "current": msg.current_battery / 100.0 if msg.current_battery != -1 else None,
        "consumed_mAh": msg.current_consumed,  # mAh
        "consumed_Wh": msg.energy_consumed,    # Wh
        "temperature": msg.temperature / 100.0 if msg.temperature != 32767 else None,
        "battery_id": msg.id,
        "battery_type": msg.type,
    }


while True:
    imu = read_imu()
    attitude = read_attitude()
    depth = read_depth()
    gps = read_gps()
    battery = read_battery()

    if imu:
        print("IMU:", imu)
    if attitude:
        print("Attitude:", attitude)
    if depth:
        print("Depth:", depth)
    if gps:
        print("GPS:", gps)
    if battery:
        print("Battery:", battery)

    print("-" * 40)
    time.sleep(1)
