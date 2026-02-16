"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)

Modified : Goran Frehse, David Filliat

Remodified : Ewen Le Hay
"""

import math
import numpy as np
import pygame
from src.path_planning_utils.simulation_parameters import parameters
from src.path_planning_utils.classes import Boat

class Slam:
    """
    Représente un slam
    """
    
    def __init__(self, Landmarks, pose, boat): # A CHANGER POUR LES PARAMETRES DE SIMULATION PARAMETERS

        self.DT = 0.1  # time tick [s]
        self.M_DIST_TH = 9.0  # Threshold of Mahalanobis distance for data association.
        self.STATE_SIZE = 3  # State size [x,y,yaw]
        self.LM_SIZE = 2  # LM state size [x,y]
        self.KNOWN_DATA_ASSOCIATION = 0  # Whether we use the true landmarks id or not

        # Simulation parameter
        # noise on control input
        self.Q_sim = (3 * np.diag([0.1, np.deg2rad(1)])) ** 2
        # noise on measurement
        self.Py_sim = (1 * np.diag([0.1, np.deg2rad(5)])) ** 2

        # Kalman filter Parameters
        # Estimated input noise for Kalman Filter
        self.Q = 2 * self.Q_sim
        # Estimated measurement noise for Kalman Filter
        self.Py = 2 * self.Py_sim
        
        # GNSS measurement noise covariance (should be tuned based on your GPS accuracy)
        # Typically GPS has ~5-10m error, so variance should reflect this
        self.R_gnss = np.diag([1.0, 1.0]) ** 2  # [x, y] position noise

        # Initial estimate of pose covariance
        self.initPEst = 0.01 * np.eye(self.STATE_SIZE)
        self.initPEst[2,2] = 0.0001  # low orientation error

        # True Landmark id for known data association
        self.trueLandmarkId =[]

        # Define landmark positions [x, y]
        self.Landmarks = Landmarks

        # Init state vector [x y yaw]' and covariance for Kalman
        self.xEst = np.zeros((self.STATE_SIZE, 1))
        self.xEst = pose
        self.PEst = self.initPEst

        # Init true state for simulator
        self.xTrue = np.zeros((self.STATE_SIZE, 1))
        self.xTrue = pose

        self.boat = boat

    # --- Helper functions

    def calc_n_lm(self, x):
        """
        Computes the number of landmarks in state vector
        """

        n = int((len(x) - self.STATE_SIZE) / self.LM_SIZE)
        return n
    
    def gps_to_local(self, lat_ref, lon_ref, north_angle, lat_target, lon_target):
        """
        Convert GPS coordinates (latitude, longitude) to local cartesian coordinates (x, y)
        relative to a reference point.
        
        Args:
            lat_ref (float): Reference latitude in degrees
            lon_ref (float): Reference longitude in degrees
            north_angle (float): Angle indicating north direction in radians (0 = +x axis)
            lat_target (float): Target latitude in degrees
            lon_target (float): Target longitude in degrees
        
        Returns:
            tuple: (dx, dy) distances in meters from reference point to target point
                   in the local coordinate frame (rotated by north_angle)
        """
        # Earth's radius in meters
        EARTH_RADIUS = 6371000.0
        
        # Convert degrees to radians
        lat_ref_rad = math.radians(lat_ref)
        lon_ref_rad = math.radians(lon_ref)
        lat_target_rad = math.radians(lat_target)
        lon_target_rad = math.radians(lon_target)
        
        # Calculate differences in latitude and longitude
        dlat = lat_target_rad - lat_ref_rad
        dlon = lon_target_rad - lon_ref_rad
        
        # Convert to local cartesian coordinates (flat Earth approximation)
        # dy: north-south distance (along latitude)
        dy_ecef = EARTH_RADIUS * dlat
        
        # dx: east-west distance (along longitude, adjusted for latitude)
        dx_ecef = EARTH_RADIUS * dlon * math.cos(lat_ref_rad)
        
        # Rotate by north_angle to align with local coordinate frame
        # north_angle is the angle of the north direction (0 = +x, CCW positive)
        cos_angle = math.cos(north_angle)
        sin_angle = math.sin(north_angle)
        
        # Rotation matrix: [cos, -sin; sin, cos]
        # Rotate from ECEF-like frame to local frame
        dx = dx_ecef * cos_angle - dy_ecef * sin_angle
        dy = dx_ecef * sin_angle + dy_ecef * cos_angle
        
        return dx, dy
    
    def calc_landmark_position(self, x, y):
        """
        Computes absolute landmark position from robot pose and observation
        """

        y_abs = np.zeros((2, 1))

        y_abs[0, 0] = x[0, 0] + y[0] * math.cos(x[2, 0] + y[1])
        y_abs[1, 0] = x[1, 0] + y[0] * math.sin(x[2, 0] + y[1])

        return y_abs


    def get_landmark_position_from_state(self, x, ind):
        """
        Extract landmark position from state vector
        """

        lm = x[self.STATE_SIZE + self.LM_SIZE * ind: self.STATE_SIZE + self.LM_SIZE * (ind + 1), :]

        return lm


    def pi_2_pi(self, angle):
        """
        Put an angle between -pi / pi
        """

        return (angle + math.pi) % (2 * math.pi) - math.pi


    def draw_polyline(self, screen, pts, color, width=2):
        if len(pts) < 2:
            return
        pygame.draw.lines(screen, color, False,
                        [(p[0], p[1]) for p in pts], width)

    def plot_covariance_ellipse_pygame(self, xEst_lim, PEst_lim, screen, color=parameters.GREEN, width=2):
        """
        Draw one covariance ellipse from covariance matrix using pygame
        Works in both static and boat-centric view modes
        """

        # Covariance position
        Pxy = PEst_lim[0:2, 0:2]

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
        cx = xEst_lim[0, 0]
        cy = xEst_lim[1, 0]

        for t in np.arange(0, 2 * math.pi + 0.1, 0.1):
            x = a * math.cos(t)
            y = b * math.sin(t)

            # Rotation
            xr = x * math.cos(angle) - y * math.sin(angle)
            yr = x * math.sin(angle) + y * math.cos(angle)

            # Translation
            px = cx + xr
            py = cy + yr

            # Translation to pygame window using get_screen_position for boat view support
            if parameters.view_type == 'boat':
                screen_pos = parameters.get_screen_position(px, py, self.boat.x, self.boat.y, self.boat.theta)
                px, py = screen_pos[0], screen_pos[1]
            else:
                px = parameters.center_offset[0] + px * parameters.scale
                py = parameters.center_offset[1] + parameters.window_height - (py) * parameters.scale

            points.append((px, py))

        # Draw ellipse
        if len(points) > 1:
            pygame.draw.lines(screen, color, True, points, width)

    def jacob_motion(self, x, u): # ça reste je pense
        """
        Compute the jacobians of motion model wrt x and u
        """

        # Jacobian of f(X,u) wrt X
        A = np.array([[1.0, 0.0, float(-self.DT * u[0,0] * math.sin(x[2, 0]))],
                    [0.0, 1.0, float(self.DT * u[0,0] * math.cos(x[2, 0]))],
                    [0.0, 0.0, 1.0]])

        # Jacobian of f(X,u) wrt u
        B = np.array([[float(self.DT * math.cos(x[2, 0])), 0.0],
                    [float(self.DT * math.sin(x[2, 0])), 0.0],
                    [0.0, self.DT]])

        return A, B

    # --- Observation model related functions

    def observation_model(self, uTrue, y=None):
        """
        Génère les observations bruyantes à partir de l'état vrai et de la commande.
        Retourne: xTrue_next, y, u_noisy

        - uTrue: control (2x1) numpy array
        - y: Seulement pour le mode temps réel, les observations sont supposées être fournies par les capteurs, donc y est une entrée de la fonction.
        """
        # helper
        def pi_2_pi(angle):
            return (angle + math.pi) % (2 * math.pi) - math.pi
        
        if parameters.realtime:
            return self.xTrue, y, uTrue # En mode temps réel, on suppose que les observations et commandes sont déjà bruitées par les capteurs et qu'on les reçoit en entrée de la fonction. On retourne juste l'état vrai pour la simulation, mais on n'utilise pas les fonctions de génération de bruit.

        else:
            # update true state
            xTrue = Boat.motion_model_np(self.xTrue, uTrue, self.DT)

            # build observations
            y = np.zeros((0, 3))

            for i in range(len(self.Landmarks[:, 0])): 
                dx = self.Landmarks[i, 0] - xTrue[0, 0]
                if self.Landmarks.size <= 2: # Quick fix pour le cas où il n'y a qu'un seul landmark
                    if i == len(self.Landmarks[:, 0]) - 1:
                        break
                    dy = self.Landmarks[i+1] - xTrue[1, 0]
                else:
                    dy = self.Landmarks[i, 1] - xTrue[1, 0]
                print(f"Landmark {i}: dx={dx}, dy={dy}")  # Debug print for landmark relative position
                d = math.hypot(dx, dy)
                angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
                if d <= self.boat.MAX_RANGE and pi_2_pi(self.boat.RANGE_YAW_DIFFERENCE[0] * math.pi / 180) <= angle and angle <= pi_2_pi(self.boat.RANGE_YAW_DIFFERENCE[1] * math.pi / 180):
                    dn = d + np.random.randn() * (self.Py_sim[0, 0] ** 0.5)
                    dn = max(dn, 0)
                    angle_n = angle + np.random.randn() * (self.Py_sim[1, 1] ** 0.5)
                    yi = np.array([dn, angle_n, i])
                    y = np.vstack((y, yi))

            # noisy input
            u = np.array([[
                uTrue[0, 0] + np.random.randn() * (self.Q_sim[0, 0] ** 0.5),
                uTrue[1, 0] + np.random.randn() * (self.Q_sim[1, 1] ** 0.5)
            ]]).T

        return xTrue, y, u

    def gnss_model(self, xGnss=None):
        """
        Simulate a GNSS measurement of the boat's position with noise.
        Returns a 2x1 numpy array [x, y].
        """
        if parameters.realtime:
            # APPEL CENTRALE GNSS
            # Ensure xGnss is properly formatted as (2x1) numpy array
            if xGnss is not None:
                xGnss = np.array(xGnss).flatten().reshape((2, 1))
            return xGnss
        else :
            # Simulate GNSS measurement with noise
            xGnss = self.xTrue[0:2] + np.random.multivariate_normal(np.zeros(2), self.R_gnss).reshape(2, 1)
            return xGnss

    def search_correspond_landmark_id(self, yi):
        """
        Landmark association with Mahalanobis distance
        """

        nLM = self.calc_n_lm(self.xEst)

        min_dist = []

        for i in range(nLM):
            innov, S, H = self.calc_innovation(yi, i)
            min_dist.append(innov.T @ np.linalg.inv(S) @ innov)
            

        min_dist.append(self.M_DIST_TH)  # new landmark

        min_id = min_dist.index(min(min_dist))

        return min_id


    def jacob_h(self, q, delta, x, i):
        """
        Compute the jacobian of observation model
        """

        sq = math.sqrt(q)
        G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                    [delta[1, 0], -delta[0, 0], -q,  -delta[1, 0], delta[0, 0]]])

        G = G / q
        nLM = self.calc_n_lm(x)
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * i)),
                        np.eye(2), np.zeros((2, 2 * nLM - 2 * (i + 1)))))

        F = np.vstack((F1, F2))

        H = G @ F

        return H


    def jacob_augment(self, x, y):
        """
        Compute the jacobians for extending covariance matrix
        """
        
        Jr = np.array([[1.0, 0.0, -y[0] * math.sin(x[2,0] + y[1])],
                    [0.0, 1.0, y[0] * math.cos(x[2,0] + y[1])]])

        Jy = np.array([[math.cos(x[2,0] + y[1]) * 10, -y[0] * math.sin(x[2,0] + y[1]) / 4],
                    [math.sin(x[2,0] + y[1]) * 10,  y[0] * math.cos(x[2,0] + y[1]) / 4]])

        return Jr, Jy


    # --- Kalman filter related functions

    def calc_innovation(self, y, LMid):
        """
        Compute innovation and Kalman gain elements for landmark observations
        """

        # Compute predicted observation from state
        lm = self.get_landmark_position_from_state(self.xEst, LMid)
        delta = lm - self.xEst[0:2] # delta = landmark position - robot position
        q = (delta.T @ delta)[0, 0] # q = delta_x^2 + delta_y^2
        y_angle = math.atan2(delta[1, 0], delta[0, 0]) - self.xEst[2, 0] # y_angle = atan2(delta_y, delta_x) - robot orientation
        yp = np.array([[math.sqrt(q), self.pi_2_pi(y_angle)]]) # yp = [range, bearing] predicted observation

        # compute innovation, i.e. diff with real observation
        innov = (y - yp).T # innovation = real observation - predicted observation
        innov[1] = self.pi_2_pi(innov[1])

        # compute matrixes for Kalman Gain
        H = self.jacob_h(q, delta, self.xEst, LMid) # H = jacobian of observation model
        S = H @ self.PEst @ H.T + self.Py
        
        return innov, S, H

    def calc_innovation_gnss(self, xGnss):
        """
        Compute innovation and Kalman gain elements for GNSS position measurement
        
        Args:
            xGnss: GNSS position measurement [x, y] as a 2x1 numpy array
        
        Returns:
            innov: innovation (measurement residual) [2x1]
            S: innovation covariance [2x2]
            H: observation matrix (Jacobian) [2x3]
        """
        # Predicted state position
        xp = self.xEst[0:2]  # [x, y]
        
        # Innovation = measured position - predicted position
        innov = xGnss - xp  # [2x1]
        
        # Jacobian of observation model for GNSS
        # h(x) = [x, y] -> dh/dx = [1, 0, 0], dh/dy = [0, 1, 0]
        # But we need to account for landmarks in state vector
        nLM = self.calc_n_lm(self.xEst)
        H = np.zeros((2, len(self.xEst)))
        H[0, 0] = 1.0  # dx/dx
        H[1, 1] = 1.0  # dy/dy
        # No dependency on yaw or landmarks
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = H @ self.PEst @ H.T + self.R_gnss
        
        return innov, S, H


    def ekf_slam(self, u, y, xGnss=None):
        """
        Apply one step of EKF predict/correct cycle
        
        Args:
            u: control input
            y: landmark observations
            xGnss: GNSS position measurement [x, y] as 2x1 numpy array (optional)
        """
        
        S = self.STATE_SIZE
        
        # Predict
        A, B = self.jacob_motion(self.xEst[0:S], u)

        self.xEst[0:S] = Boat.motion_model_np(self.xEst[0:S], u, self.DT) # use robot motion model

        self.PEst[0:S, 0:S] = A @ self.PEst[0:S, 0:S] @ A.T + B @ self.Q @ B.T
        self.PEst[0:S,S:] = A @ self.PEst[0:S,S:]
        self.PEst[S:,0:S] = self.PEst[0:S,S:].T

        self.PEst = (self.PEst + self.PEst.T) / 2.0  # ensure symetry
        
        # Update with landmark observations
        for iy in range(len(y[:, 0])):  # for each observation
            nLM = self.calc_n_lm(self.xEst)
            
            if self.KNOWN_DATA_ASSOCIATION:
                try:
                    min_id = self.trueLandmarkId.index(y[iy, 2])
                except ValueError:
                    min_id = nLM
                    self.trueLandmarkId.append(y[iy, 2])
            else:
                min_id = self.search_correspond_landmark_id(y[iy, 0:2])

            # Extend map if required
            if min_id == nLM:
                print("New LM")

                # Regular
                # Extend state and covariance matrix
                self.xEst = np.vstack((self.xEst, self.calc_landmark_position(self.xEst, y[iy, :])))

                Jr, Jy = self.jacob_augment(self.xEst[0:3], y[iy, :])
                bottomPart = np.hstack((Jr @ self.PEst[0:3, 0:3], Jr @ self.PEst[0:3, 3:]))
                rightPart = bottomPart.T
                self.PEst = np.vstack((np.hstack((self.PEst, rightPart)),
                                np.hstack((bottomPart,
                                Jr @ self.PEst[0:3, 0:3] @ Jr.T + Jy @ self.Py @ Jy.T))))

            else:
                # Perform Kalman update for landmark
                innov, S, H = self.calc_innovation(y[iy, 0:2], min_id)
                K = (self.PEst @ H.T) @ np.linalg.inv(S)
                
                self.xEst = self.xEst + (K @ innov)
                            
                self.PEst = (np.eye(len(self.xEst)) - K @ H) @ self.PEst
                self.PEst = 0.5 * (self.PEst + self.PEst.T)  # Ensure symetry

        # Update with GNSS measurement if available
        if xGnss is not None:
            innov_gnss, S_gnss, H_gnss = self.calc_innovation_gnss(xGnss)
            K_gnss = (self.PEst @ H_gnss.T) @ np.linalg.inv(S_gnss)
            
            self.xEst = self.xEst + (K_gnss @ innov_gnss)
            
            self.PEst = (np.eye(len(self.xEst)) - K_gnss @ H_gnss) @ self.PEst
            self.PEst = 0.5 * (self.PEst + self.PEst.T)  # Ensure symetry
            
        self.xEst[2] = self.pi_2_pi(self.xEst[2])

        return self.xEst, self.PEst


    def get_estimate_full_motion(self, uTrue, y_rt=None, xGnss_rt=None):
        # Simulate motion and generate u and y
        self.xTrue, y, u =  self.observation_model(uTrue, y_rt)
        xGnss = self.gnss_model(xGnss_rt)
        self.xEst, self.PEst = self.ekf_slam(u, y, xGnss)

        return self.xEst, self.PEst

    def show_animation(self, screen):

        """
        Only plot things related to the slam, we suppose that the screen is set accordingly (reseted in a loop, with other plots)
        """

        # Show range
        if not parameters.realtime:
            self.boat.x_true, self.boat.y_true, self.boat.theta_true = self.xTrue[0,0], self.xTrue[1,0], self.xTrue[2,0]
            self.boat.show_true(screen)
        self.boat.show_range(screen)

        # Landmarks supposés déjà affichés
        # for lm in self.Landmarks:
        #     pygame.draw.circle(screen, (0, 0, 0),
        #                     (parameters.center_offset[0] + lm[0] * parameters.scale, parameters.center_offset[1] + parameters.window_height - (lm[1]) * parameters.scale), 5)

        # Pose estimée
        pygame.draw.circle(screen, parameters.DARK_GREEN,
                        (parameters.center_offset[0] + self.xEst[0, 0] * parameters.scale, parameters.center_offset[1] + parameters.window_height - (self.xEst[1, 0]) * parameters.scale), 5)

        self.plot_covariance_ellipse_pygame(
            self.xEst[0:self.STATE_SIZE],
            self.PEst[0:self.STATE_SIZE, 0:self.STATE_SIZE],
            screen,
            color=parameters.DARK_GREEN
        )

        # Landmarks estimés
        for i in range(self.calc_n_lm(self.xEst)):
            idx = self.STATE_SIZE + i * 2
            pygame.draw.line(
                screen, parameters.DARK_GREEN,
                (parameters.center_offset[0] + (self.xEst[idx, 0] - 0.2) * parameters.scale, parameters.center_offset[1] + parameters.window_height - (self.xEst[idx + 1, 0] - 0.2) * parameters.scale),
                (parameters.center_offset[0] + (self.xEst[idx, 0] + 0.2) * parameters.scale, parameters.center_offset[1] + parameters.window_height - (self.xEst[idx + 1, 0] + 0.2) * parameters.scale), 2
            )
            pygame.draw.line(
                screen, parameters.DARK_GREEN,
                (parameters.center_offset[0] + (self.xEst[idx, 0] - 0.2) * parameters.scale, parameters.center_offset[1] + parameters.window_height - (self.xEst[idx + 1, 0] + 0.2) * parameters.scale),
                (parameters.center_offset[0] + (self.xEst[idx, 0] + 0.2) * parameters.scale, parameters.center_offset[1] + parameters.window_height - (self.xEst[idx + 1, 0] - 0.2) * parameters.scale), 2
            )

            self.plot_covariance_ellipse_pygame(
                self.xEst[idx:idx + 2],
                self.PEst[idx:idx + 2, idx:idx + 2],
                screen,
                color=parameters.DARK_GREEN
            )
