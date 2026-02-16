class State:
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle (heading)
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super().__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle using the bicycle model.

        The bicycle model approximates the vehicle as having front and rear wheels
        on a single axis, with steering only at the front.

        :param acceleration: (float) Longitudinal acceleration
        :param delta: (float) Steering angle
        """
        # Limit steering angle to maximum
        delta = np.clip(delta, -max_steer, max_steer)

        # Update position using current velocity and heading
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        # Update heading using bicycle model kinematics
        self.yaw += self.v / L * np.tan(delta) * dt
        # Normalize yaw to [-pi, pi]
        self.yaw = normalize_angle(self.yaw)
        # Update speed
        self.v += acceleration * dt