"""
Cubic spline planner

This module provides classes for 1D and 2D cubic spline interpolation.
Cubic splines are piecewise cubic polynomials that ensure smooth curves
passing through given data points, with continuous first and second derivatives.

Author: Atsushi Sakai(@Atsushi_twi)

Classes:
- CubicSpline1D: For interpolating 1D data (e.g., y as function of x)
- CubicSpline2D: For generating smooth 2D paths from waypoints, parameterized by arc length

Functions:
- calc_spline_course: Convenience function to generate a full path with yaw and curvature
"""
import math
import numpy as np
import bisect


class CubicSpline1D:
    """
    1D Cubic Spline class

    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted
        in ascending order.
    y : list
        y coordinates for data points

    Examples
    --------
    You can interpolate 1D data points.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(5)
    >>> y = [1.7, -6, 5, 6.5, 0.0]
    >>> sp = CubicSpline1D(x, y)
    >>> xi = np.linspace(0.0, 5.0)
    >>> yi = [sp.calc_position(x) for x in xi]
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(xi, yi , "r", label="Cubic spline interpolation")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_1d.png

    """

    def __init__(self, x, y):

        # Calculate the differences between consecutive x points (intervals)
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        # Initialize lists to store spline coefficients
        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x  # x coordinates of data points
        self.y = y  # y coordinates of data points
        self.nx = len(x)  # Number of data points

        # Coefficient a is simply the y values at each point
        self.a = [iy for iy in y]

        # Calculate coefficient c by solving the linear system A*c = B
        A = self.__calc_A(h)  # Matrix A for the system
        B = self.__calc_B(h, self.a)  # Vector B for the system
        self.c = np.linalg.solve(A, B)  # Solve for c

        # Calculate coefficients b and d for each interval
        for i in range(self.nx - 1):
            # d_i = (c_{i+1} - c_i) / (3 * h_i)
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            # b_i = (a_{i+1} - a_i)/h_i - h_i/3 * (2*c_i + c_{i+1})
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x` range, return None.

        Parameters
        ----------
        x : float
            x position to calculate y.

        Returns
        -------
        y : float
            y position for given x.
        """
        # Check if x is within the range of data points
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        # Find the interval index where x lies
        i = self.__search_index(x)
        # Calculate the distance from the start of the interval
        dx = x - self.x[i]
        # Evaluate the cubic polynomial: y = a + b*dx + c*dx^2 + d*dx^3
        position = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.

        if x is outside the input x, return None

        Parameters
        ----------
        x : float
            x position to calculate first derivative.

        Returns
        -------
        dy : float
            first derivative for given x.
        """

        # Check bounds
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        # Find interval
        i = self.__search_index(x)
        dx = x - self.x[i]
        # First derivative: dy/dx = b + 2*c*dx + 3*d*dx^2
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.

        if x is outside the input x, return None

        Parameters
        ----------
        x : float
            x position to calculate second derivative.

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        # Check bounds
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        # Find interval
        i = self.__search_index(x)
        dx = x - self.x[i]
        # Second derivative: d²y/dx² = 2*c + 6*d*dx
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def calc_third_derivative(self, x):
        """
        Calc third derivative at given x.

        if x is outside the input x, return None

        Parameters
        ----------
        x : float
            x position to calculate third derivative.

        Returns
        -------
        dddy : float
            third derivative for given x.
        """
        # Check bounds
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        # Find interval
        i = self.__search_index(x)
        # Third derivative: d³y/dx³ = 6*d (constant in each interval)
        dddy = 6.0 * self.d[i]
        return dddy

    def __search_index(self, x):
        """
        Find the index of the interval where x lies using binary search.
        Returns the index i such that x[i] <= x < x[i+1]
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        Calculate the tridiagonal matrix A for solving the spline coefficients c.
        This matrix enforces continuity of first and second derivatives.
        """
        A = np.zeros((self.nx, self.nx))
        # Boundary conditions: second derivative at endpoints is zero
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                # Diagonal elements: 2*(h[i] + h[i+1])
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            # Off-diagonal elements
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        # Set boundary conditions explicitly
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        Calculate the vector B for the linear system A*c = B.
        B represents the second derivatives based on the data points.
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            # B[i+1] = 3 * [(a[i+2] - a[i+1])/h[i+1] - (a[i+1] - a[i])/h[i]]
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1]\
                - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class

    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.

    Examples
    --------
    You can interpolate a 2D data points.

    >>> import matplotlib.pyplot as plt
    >>> x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    >>> y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    >>> ds = 0.1  # [m] distance of each interpolated points
    >>> sp = CubicSpline2D(x, y)
    >>> s = np.arange(0, sp.s[-1], ds)
    >>> rx, ry, ryaw, rk = [], [], [], []
    >>> for i_s in s:
    ...     ix, iy = sp.calc_position(i_s)
    ...     rx.append(ix)
    ...     ry.append(iy)
    ...     ryaw.append(sp.calc_yaw(i_s))
    ...     rk.append(sp.calc_curvature(i_s))
    >>> plt.subplots(1)
    >>> plt.plot(x, y, "xb", label="Data points")
    >>> plt.plot(rx, ry, "-r", label="Cubic spline path")
    >>> plt.grid(True)
    >>> plt.axis("equal")
    >>> plt.xlabel("x[m]")
    >>> plt.ylabel("y[m]")
    >>> plt.legend()
    >>> plt.show()

    .. image:: cubic_spline_2d_path.png

    >>> plt.subplots(1)
    >>> plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("yaw angle[deg]")

    .. image:: cubic_spline_2d_yaw.png

    >>> plt.subplots(1)
    >>> plt.plot(s, rk, "-r", label="curvature")
    >>> plt.grid(True)
    >>> plt.legend()
    >>> plt.xlabel("line length[m]")
    >>> plt.ylabel("curvature [1/m]")

    .. image:: cubic_spline_2d_curvature.png
    """

    def __init__(self, x, y):
        # Calculate the cumulative arc length s along the path
        self.s = self.__calc_s(x, y)
        # Create 1D splines for x and y as functions of s
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        """
        Calculate the cumulative arc length s for parameterization.
        s[i] is the total distance from start to point i.
        """
        # Differences in x and y
        dx = np.diff(x)
        dy = np.diff(y)
        # Distance between consecutive points
        self.ds = np.hypot(dx, dy)
        # Cumulative sum of distances
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        # Interpolate x and y at arc length s
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        # First derivatives (velocity components)
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        # Second derivatives (acceleration components)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        # Curvature formula: k = (ddy*dx - ddx*dy) / (dx^2 + dy^2)^{3/2}
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_curvature_rate(self, s):
        """
        calc curvature rate

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature rate for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        a = dx * ddy - dy * ddx
        b = dx * dddy - dy * dddx
        c = dx * ddx + dy * ddy
        d = dx * dx + dy * dy
        return (b * d - 3.0 * a * c) / (d * d * d)

    def calc_yaw(self, s):
        """
        calc yaw

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        # First derivatives give the direction vector
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        # Yaw is the angle of the tangent vector
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    """
    Generate a smooth path from waypoints using cubic splines.

    Parameters
    ----------
    x : list
        x coordinates of waypoints.
    y : list
        y coordinates of waypoints.
    ds : float
        Step size for arc length parameterization.

    Returns
    -------
    rx : list
        Interpolated x positions.
    ry : list
        Interpolated y positions.
    ryaw : list
        Yaw angles at each point.
    rk : list
        Curvatures at each point.
    s : list
        Arc lengths.
    """
    # Create 2D spline from waypoints
    sp = CubicSpline2D(x, y)
    # Generate arc length values from 0 to total length
    s = list(np.arange(0, sp.s[-1], ds))

    # Interpolate position, yaw, and curvature at each s
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def main_1d():
    """
    Test function for 1D cubic spline interpolation.
    Plots the original data points and the interpolated curve.
    """
    print("CubicSpline1D test")
    import matplotlib.pyplot as plt
    # Sample data points
    x = np.arange(5)
    y = [1.7, -6, 5, 6.5, 0.0]
    # Create spline
    sp = CubicSpline1D(x, y)
    # Interpolation points
    xi = np.linspace(0.0, 5.0)

    # Plot
    plt.plot(x, y, "xb", label="Data points")
    plt.plot(xi, [sp.calc_position(x) for x in xi], "r",
             label="Cubic spline interpolation")
    plt.grid(True)
    plt.legend()
    plt.show()


def main_2d():  # pragma: no cover
    """
    Test function for 2D cubic spline path planning.
    Plots the path, yaw, and curvature.
    """
    print("CubicSpline1D 2D test")
    import matplotlib.pyplot as plt
    # Sample waypoints
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    ds = 0.1  # [m] distance of each interpolated points

    # Create 2D spline
    sp = CubicSpline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    # Interpolate
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    # Plot path
    plt.subplots(1)
    plt.plot(x, y, "xb", label="Data points")
    plt.plot(rx, ry, "-r", label="Cubic spline path")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    # Plot yaw
    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    # Plot curvature
    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()


if __name__ == '__main__':
    # main_1d()
    main_2d()
