"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)

Modified : Goran Frehse, David Filliat

Remodified : Ewen Le Hay
"""

import math
import numpy as np
from pygame_manager import *

DT = 0.1  # time tick [s]
SIM_TIME = 60.0  # simulation time [s] # Shortened for undelayed initialisation of landmarks
MAX_RANGE = 10.0  # maximum observation range
M_DIST_TH = 9.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
KNOWN_DATA_ASSOCIATION = 0  # Whether we use the true landmarks id or not


# --- Motion model related functions -> A changer !!!!!!!!!!

def calc_input():
    """
    Generate a control vector to make the robot follow a circular trajectory
    """

    v = 1  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def motion_model(x, u):
    """
    Compute future robot position from current position and control
    """
    
    xp = np.array([[x[0,0] + u[0,0]*DT * math.cos(x[2,0])],
                  [x[1,0] + u[0,0]*DT * math.sin(x[2,0])],
                  [x[2,0] + u[1,0]*DT]])
    xp[2] = pi_2_pi(xp[2])

    return xp.reshape((3, 1))


def jacob_motion(x, u): # ça reste je pense
    """
    Compute the jacobians of motion model wrt x and u
    """

    # Jacobian of f(X,u) wrt X
    A = np.array([[1.0, 0.0, float(-DT * u[0,0] * math.sin(x[2, 0]))],
                  [0.0, 1.0, float(DT * u[0,0] * math.cos(x[2, 0]))],
                  [0.0, 0.0, 1.0]])

    # Jacobian of f(X,u) wrt u
    B = np.array([[float(DT * math.cos(x[2, 0])), 0.0],
                  [float(DT * math.sin(x[2, 0])), 0.0],
                  [0.0, DT]])

    return A, B


# Simulation parameter
# noise on control input
Q_sim = (3 * np.diag([0.1, np.deg2rad(1)])) ** 2
# noise on measurement
Py_sim = (1 * np.diag([0.1, np.deg2rad(5)])) ** 2

# Kalman filter Parameters
# Estimated input noise for Kalman Filter
Q = 2 * Q_sim
# Estimated measurement noise for Kalman Filter
Py = 2 * Py_sim

# Initial estimate of pose covariance
initPEst = 0.01 * np.eye(STATE_SIZE)
initPEst[2,2] = 0.0001  # low orientation error

# True Landmark id for known data association
trueLandmarkId =[]

# --- Helper functions

def calc_n_lm(x):
    """
    Computes the number of landmarks in state vector
    """

    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def calc_landmark_position(x, y):
    """
    Computes absolute landmark position from robot pose and observation
    """

    y_abs = np.zeros((2, 1))

    y_abs[0, 0] = x[0, 0] + y[0] * math.cos(x[2, 0] + y[1])
    y_abs[1, 0] = x[1, 0] + y[0] * math.sin(x[2, 0] + y[1])

    return y_abs


def get_landmark_position_from_state(x, ind): # important ?
    """
    Extract landmark position from state vector
    """

    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def pi_2_pi(angle):
    """
    Put an angle between -pi / pi
    """

    return (angle + math.pi) % (2 * math.pi) - math.pi


def draw_polyline(screen, pts, color, width=2):
    if len(pts) < 2:
        return
    pygame.draw.lines(screen, color, False,
                      [(p[0], p[1]) for p in pts], width)

def plot_covariance_ellipse_pygame(xEst, PEst, screen, color=(255, 0, 0), width=2):
    """
    Draw one covariance ellipse from covariance matrix using pygame
    """

    # Covariance position
    Pxy = PEst[0:2, 0:2]

    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    if eigval[smallind] < 0:
        print("Problem with Pxy:\n", Pxy)
        return

    # Angle of ellipse
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])

    # Ellipse radii (3σ)
    a = 3.0 * math.sqrt(eigval[bigind])
    b = 3.0 * math.sqrt(eigval[smallind])

    # Generate ellipse points
    points = []
    cx = xEst[0, 0]
    cy = xEst[1, 0]

    for t in np.arange(0, 2 * math.pi + 0.1, 0.1):
        x = a * math.cos(t)
        y = b * math.sin(t)

        # Rotation
        xr = x * math.cos(angle) - y * math.sin(angle)
        yr = x * math.sin(angle) + y * math.cos(angle)

        # Translation
        px = cx + xr
        py = cy + yr

        # Translation to pygame window
        px = center_offset[0] + px * scale
        py = center_offset[1] + window_height - (py) * scale

        points.append((px, py))

    # Draw ellipse
    if len(points) > 1:
        pygame.draw.lines(screen, color, True, points, width)

# --- Observation model related functions

def observation(xTrue, xd, uTrue, Landmarks):
    """
    Generate noisy control and observation and update true position and dead reckoning
    """
    xTrue = motion_model(xTrue, uTrue)

    # add noise to gps x-y
    y = np.zeros((0, 3))

    for i in range(len(Landmarks[:, 0])):

        dx = Landmarks[i, 0] - xTrue[0, 0]
        dy = Landmarks[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Py_sim[0, 0] ** 0.5  # add noise
            dn = max(dn,0)
            angle_n = angle + np.random.randn() * Py_sim[1, 1] ** 0.5  # add noise
            yi = np.array([dn, angle_n, i])
            y = np.vstack((y, yi))

    # add noise to input
    u = np.array([[
        uTrue[0, 0] + np.random.randn() * Q_sim[0, 0] ** 0.5,
        uTrue[1, 0] + np.random.randn() * Q_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, u)
    
    return xTrue, y, xd, u


def search_correspond_landmark_id(xEst, PEst, yi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xEst)

    min_dist = []

    for i in range(nLM):
        innov, S, H = calc_innovation(xEst, PEst, yi, i)
        min_dist.append(innov.T @ np.linalg.inv(S) @ innov)
        

    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def jacob_h(q, delta, x, i):
    """
    Compute the jacobian of observation model
    """

    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], -delta[0, 0], -q,  -delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * i)),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * (i + 1)))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def jacob_augment(x, y):
    """
    Compute the jacobians for extending covariance matrix
    """
    
    Jr = np.array([[1.0, 0.0, -y[0] * math.sin(x[2,0] + y[1])],
                   [0.0, 1.0, y[0] * math.cos(x[2,0] + y[1])]])

    Jy = np.array([[math.cos(x[2,0] + y[1]) * 10, -y[0] * math.sin(x[2,0] + y[1]) / 4],
                   [math.sin(x[2,0] + y[1]) * 10,  y[0] * math.cos(x[2,0] + y[1]) / 4]])

    return Jr, Jy


# --- Kalman filter related functions

def calc_innovation(xEst, PEst, y, LMid):
    """
    Compute innovation and Kalman gain elements
    """

    # Compute predicted observation from state
    lm = get_landmark_position_from_state(xEst, LMid)
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    y_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    yp = np.array([[math.sqrt(q), pi_2_pi(y_angle)]])

    # compute innovation, i.e. diff with real observation
    innov = (y - yp).T
    innov[1] = pi_2_pi(innov[1])

    # compute matrixes for Kalman Gain
    H = jacob_h(q, delta, xEst, LMid)
    S = H @ PEst @ H.T + Py
    
    return innov, S, H


def ekf_slam(xEst, PEst, u, y):
    """
    Apply one step of EKF predict/correct cycle
    """
    
    S = STATE_SIZE
    
    # Predict
    A, B = jacob_motion(xEst[0:S], u)

    xEst[0:S] = motion_model(xEst[0:S], u)

    PEst[0:S, 0:S] = A @ PEst[0:S, 0:S] @ A.T + B @ Q @ B.T
    PEst[0:S,S:] = A @ PEst[0:S,S:]
    PEst[S:,0:S] = PEst[0:S,S:].T

    PEst = (PEst + PEst.T) / 2.0  # ensure symetry
    
    # Update
    for iy in range(len(y[:, 0])):  # for each observation
        nLM = calc_n_lm(xEst)
        
        if KNOWN_DATA_ASSOCIATION:
            try:
                min_id = trueLandmarkId.index(y[iy, 2])
            except ValueError:
                min_id = nLM
                trueLandmarkId.append(y[iy, 2])
        else:
            min_id = search_correspond_landmark_id(xEst, PEst, y[iy, 0:2])

        # Extend map if required
        if min_id == nLM:
            print("New LM")

            # # Add undelayed initialisation of landmarks
            # """
            # Add 3 landmarks here
            # """
            # new_landmarks = calc_landmark_position_direction(xEst, y[iy, :])
            # for i in range (3):
            #     xEst = np.vstack((xEst, new_landmarks[i]))

            #     Jr, Jy = jacob_augment(xEst[0:3], y[iy, :])
            #     bottomPart = np.hstack((Jr @ PEst[0:3, 0:3], Jr @ PEst[0:3, 3:])) 
            #     rightPart = bottomPart.T
            #     PEst = np.vstack((np.hstack((PEst, rightPart)),
            #                     np.hstack((bottomPart,
            #                     Jr @ PEst[0:3, 0:3] @ Jr.T + Jy @ Py @ Jy.T))))

            # Regular
            # Extend state and covariance matrix
            xEst = np.vstack((xEst, calc_landmark_position(xEst, y[iy, :])))

            Jr, Jy = jacob_augment(xEst[0:3], y[iy, :])
            bottomPart = np.hstack((Jr @ PEst[0:3, 0:3], Jr @ PEst[0:3, 3:]))
            rightPart = bottomPart.T
            PEst = np.vstack((np.hstack((PEst, rightPart)),
                              np.hstack((bottomPart,
                              Jr @ PEst[0:3, 0:3] @ Jr.T + Jy @ Py @ Jy.T))))

        else:
            # Perform Kalman update
            innov, S, H = calc_innovation(xEst, PEst, y[iy, 0:2], min_id)
            K = (PEst @ H.T) @ np.linalg.inv(S)
            
            xEst = xEst + (K @ innov)
                        
            PEst = (np.eye(len(xEst)) - K @ H) @ PEst
            PEst = 0.5 * (PEst + PEst.T)  # Ensure symetry

            """
            Remove the landmark if xSTD is too large
            """ 
            
            # xSTD = np.sqrt(np.diag(PEst)).T # Column vector of standard deviations (sqrt of diagonal of PEst)

            # if (xSTD[min_id*3] > 5 or xSTD[min_id*3 + 1] > 5):
            #     # remove landmark
            #     PEst = np.delete(np.delete(PEst, min_id, axis=0), min_id, axis=1)
            #     PEst = np.delete(np.delete(PEst, min_id, axis=0), min_id, axis=1)
            #     xEst = np.delete(xEst, min_id, axis=0)
            #     xEst = np.delete(xEst, min_id, axis=0)



        
    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


# --- Main script

def main():
    print(__file__ + " start!!")

    time = 0.0

    # Define landmark positions [x, y]

    # Dense small loop (yaw_rate = 0.5)
    # Landmarks = np.array([[0.0, 5.0],
    #                       [5.0, 1.0],
    #                       [3.0, 4.0],
    #                       [-5.0, 0.0],
    #                       [-2.0, 1.0],
    #                       [1.0, -1.0],
    #                       [-4.0, 4.0],
    #                       [2.0, 2.0],
    #                       [0.0, 1.5],
    #                       [-0.5, 0.5]])
    
    # Sparse large loop (yaw_rate = 0.1)
    # Landmarks = np.array([[-5.0, 0.0],
    #                     [2.0, 2.0],
    #                     [-0.5, 0.5]])

    # Dense large loop (yaw_rate = 0.1)
    Landmarks = np.array([[5.0, 5.0],
                        [5.0, 1.0],
                        [10.0, 15.0],
                        [-5.0, 0.0],
                        [-2.0, 1.0],
                        [1.0, -1.0],
                        [-4.0, 4.0],
                        [11.0, 10.0],
                        [0.0, 1.5],
                        [-6, 20.0],
                        [-10, 15],
                        [0, 21],
                        [3, 18],
                        [-2, 17],
                        [-8, 8],
                        [9, 7],
                        [-11, 12],
                        [6, 14],
                        [-10, 5]])

    # Lone landmark for undelayed initialisation tests
    # Landmarks = np.array([[5.0, 10.0]])

    # Init state vector [x y yaw]' and covariance for Kalman
    xEst = np.zeros((STATE_SIZE, 1))
    xEst[2, 0] = 0  #np.pi/2 # Face north
    PEst = initPEst

    # Init true state for simulator
    xTrue = np.zeros((STATE_SIZE, 1))
    xTrue[2, 0] = 0 #np.pi/2 # Face north

    # Init dead reckoning (sum of individual controls)
    xDR = np.zeros((STATE_SIZE, 1))
    xDR[2, 0] = 0 #np.pi/2 # Face north

    # Init history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hxError = np.abs(xEst-xTrue)  # pose error
    hxVar = np.sqrt(np.diag(PEst[0:STATE_SIZE,0:STATE_SIZE]).reshape(3,1))  #state std dev


    # counter for plotting
    count = 0

    while  time <= SIM_TIME:
        count = count + 1
        time += DT

        # Simulate motion and generate u and y
        uTrue = calc_input()
        xTrue, y, xDR, u = observation(xTrue, xDR, uTrue, Landmarks)

        xEst, PEst = ekf_slam(xEst, PEst, u, y)

        # store data history
        hxEst = np.hstack((hxEst, xEst[0:STATE_SIZE]))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        err = xEst[0:STATE_SIZE]-xTrue
        err[2] = pi_2_pi(err[2])
        hxError = np.hstack((hxError,err))
        hxVar = np.hstack((hxVar,np.sqrt(np.diag(PEst[0:STATE_SIZE,0:STATE_SIZE]).reshape(3,1))))


        if show_animation and count % 15 == 0:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            screen.fill((255, 255, 255))
            draw_basic_screen(screen)

            # Show range
            radius = MAX_RANGE*scale
            circle = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(circle, (255, 255, 0, 128), (radius, radius), radius)
            center = (center_offset[0] + (xTrue[0, 0]) * scale - radius, center_offset[1] + window_height - (xTrue[1, 0]) * scale - radius)
            # - radius + radius
            screen.blit(circle, center)

            # Landmarks
            for lm in Landmarks:
                pygame.draw.circle(screen, (0, 0, 0),
                                (center_offset[0] + lm[0] * scale, center_offset[1] + window_height - (lm[1]) * scale), 5)

            # Pygame adaptation
            newhxTrue = np.array([center_offset[0] + hxTrue[0, :] * scale, center_offset[1] + window_height - (hxTrue[1, :]) * scale, hxTrue[2, :]])
            newhxDR = np.array([center_offset[0] + hxDR[0, :] * scale, center_offset[1] + window_height - (hxDR[1, :]) * scale, hxDR[2, :]])
            newhxEst = np.array([center_offset[0] + hxEst[0, :] * scale, center_offset[1] + window_height - (hxEst[1, :]) * scale, hxEst[2, :]])

            # Trajectoires
            draw_polyline(screen, newhxTrue.T, (0, 0, 0), 2)      # True
            draw_polyline(screen, newhxDR.T, (0, 200, 0), 2)      # Odom
            draw_polyline(screen, newhxEst.T, (200, 0, 0), 2)     # EKF

            # Pose estimée
            pygame.draw.circle(screen, (200, 0, 0),
                            (center_offset[0] + xEst[0, 0] * scale, center_offset[1] + window_height - (xEst[1, 0]) * scale), 5)

            plot_covariance_ellipse_pygame(
                xEst[0:STATE_SIZE],
                PEst[0:STATE_SIZE, 0:STATE_SIZE],
                screen
            )

            # Landmarks estimés
            for i in range(calc_n_lm(xEst)):
                idx = STATE_SIZE + i * 2
                pygame.draw.line(
                    screen, (200, 0, 0),
                    (center_offset[0] + (xEst[idx, 0] - 0.2) * scale, center_offset[1] + window_height - (xEst[idx + 1, 0] - 0.2) * scale),
                    (center_offset[0] + (xEst[idx, 0] + 0.2) * scale, center_offset[1] + window_height - (xEst[idx + 1, 0] + 0.2) * scale), 2
                )
                pygame.draw.line(
                    screen, (200, 0, 0),
                    (center_offset[0] + (xEst[idx, 0] - 0.2) * scale, center_offset[1] + window_height - (xEst[idx + 1, 0] + 0.2) * scale),
                    (center_offset[0] + (xEst[idx, 0] + 0.2) * scale, center_offset[1] + window_height - (xEst[idx + 1, 0] - 0.2) * scale), 2
                )

                plot_covariance_ellipse_pygame(
                    xEst[idx:idx + 2],
                    PEst[idx:idx + 2, idx:idx + 2],
                    screen
                )

            pygame.display.flip()
            clock.tick(60)

    # Wait for user to close window
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False

    pygame.quit() 

    tErrors = np.sqrt(np.square(hxError[0, :]) + np.square(hxError[1, :]))
    oErrors = np.sqrt(np.square(hxError[2, :]))
    print("Mean (var) translation error : {:e} ({:e})".format(np.mean(tErrors), np.var(tErrors)))
    print("Mean (var) rotation error : {:e} ({:e})".format(np.mean(oErrors), np.var(oErrors)))    # keep window open

if __name__ == '__main__':
    main()
