"""
Example demonstrating GNSS integration in EKF-SLAM
Use this to test the GNSS functionality
"""

import numpy as np
import matplotlib.pyplot as plt

# Hypothetical example (adapt to your actual classes)
# from ekf_slam import Slam
# from boat_state import Boat

def example_gnss_slam():
    """
    Example of how to use GNSS with EKF-SLAM
    """
    
    # Setup
    print("=" * 60)
    print("GNSS-SLAM Integration Example")
    print("=" * 60)
    
    # Define landmarks
    landmarks = np.array([
        [10, 10],
        [20, 5],
        [15, 20],
        [5, 15]
    ])
    
    # Initial pose [x, y, yaw]
    initial_pose = np.array([[0.0], [0.0], [0.0]])
    
    print("\n1. SLAM Initialization")
    print("-" * 40)
    print(f"Landmarks: {landmarks.shape[0]} detected")
    print(f"Initial pose: ({initial_pose[0,0]:.2f}, {initial_pose[1,0]:.2f})")
    
    # Configuration examples
    print("\n2. GNSS Configuration")
    print("-" * 40)
    
    # Scenario 1: High precision GPS (0.5m error)
    print("\nScenario 1: High precision GPS")
    R_gnss_high = np.diag([0.5, 0.5]) ** 2
    print(f"R_gnss = {R_gnss_high[0,0]:.3f}")
    print("→ Use this if your GPS is RTK-corrected or similar")
    
    # Scenario 2: Standard GPS (2m error)
    print("\nScenario 2: Standard GPS (typical)")
    R_gnss_standard = np.diag([2.0, 2.0]) ** 2
    print(f"R_gnss = {R_gnss_standard[0,0]:.3f}")
    print("→ Use this for standard consumer GPS")
    
    # Scenario 3: Poor GPS (5m error)
    print("\nScenario 3: Poor GPS (degraded reception)")
    R_gnss_poor = np.diag([5.0, 5.0]) ** 2
    print(f"R_gnss = {R_gnss_poor[0,0]:.3f}")
    print("→ Use this when GPS signal is weak")
    
    # Usage examples
    print("\n3. Usage Pattern")
    print("-" * 40)
    
    print("\nPattern A: Always use GNSS")
    print("""
    slam.USE_GNSS = True
    slam.R_gnss = np.diag([2.0, 2.0]) ** 2
    
    xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
    """)
    
    print("\nPattern B: Conditional GNSS (safer)")
    print("""
    if gnss_available and gnss_signal_quality > THRESHOLD:
        xEst, PEst = slam.get_estimate_full_motion(u, xGnss)
    else:
        # Fallback to landmarks only
        xEst, PEst = slam.get_estimate_full_motion(u, xGnss=None)
    """)
    
    print("\nPattern C: Adaptive trust level")
    print("""
    # Start with low trust in GPS
    slam.R_gnss = np.diag([5.0, 5.0]) ** 2
    
    # Monitor innovation (measurement residual)
    innovation = z_gnss - h(x_est)
    
    # Increase trust if innovation is small
    if np.linalg.norm(innovation) < THRESHOLD:
        slam.R_gnss *= 0.95  # Increase trust
    else:
        slam.R_gnss *= 1.05  # Decrease trust
    """)
    
    # Simulation example
    print("\n4. Simulation Example")
    print("-" * 40)
    
    # Simulate a trajectory with measurements
    num_steps = 20
    true_trajectory = np.zeros((3, num_steps + 1))
    gnss_measurements = np.zeros((2, num_steps + 1))
    estimated_trajectory = np.zeros((3, num_steps + 1))
    
    # Simulate
    for step in range(num_steps):
        # True state evolves
        v = 1.0  # velocity
        omega = 0.1  # angular velocity
        dt = 0.1
        
        x, y, theta = true_trajectory[:, step]
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + omega * dt
        
        true_trajectory[:, step + 1] = [x_next, y_next, theta_next]
        
        # GNSS measurement (with noise)
        gnss_noise_x = np.random.randn() * 2.0  # 2m standard deviation
        gnss_noise_y = np.random.randn() * 2.0
        
        gnss_measurements[:, step + 1] = [
            x_next + gnss_noise_x,
            y_next + gnss_noise_y
        ]
        
        # Estimated state (simplified: GNSS + small filter error)
        filter_error = np.random.randn(3) * 0.1
        estimated_trajectory[:, step + 1] = true_trajectory[:, step + 1] + filter_error
    
    # Results
    print("\nSimulation Results (first 5 steps):")
    print("Step | True State        | GNSS Measurement  | Estimated State")
    print("-" * 70)
    for step in range(min(5, num_steps + 1)):
        x_true, y_true, th_true = true_trajectory[:, step]
        x_gnss, y_gnss = gnss_measurements[:, step]
        x_est, y_est, th_est = estimated_trajectory[:, step]
        
        print(f"{step:4d} | ({x_true:6.2f},{y_true:6.2f}) | "
              f"({x_gnss:6.2f},{y_gnss:6.2f}) | "
              f"({x_est:6.2f},{y_est:6.2f})")
    
    # Statistical analysis
    print("\n5. Error Analysis")
    print("-" * 40)
    
    gnss_error = np.linalg.norm(
        true_trajectory[0:2, :] - gnss_measurements[0:2, :],
        axis=0
    )
    est_error = np.linalg.norm(
        true_trajectory[0:2, :] - estimated_trajectory[0:2, :],
        axis=0
    )
    
    print(f"\nGNSS Error Statistics:")
    print(f"  Mean: {np.mean(gnss_error):.3f} m")
    print(f"  Std:  {np.std(gnss_error):.3f} m")
    print(f"  Max:  {np.max(gnss_error):.3f} m")
    
    print(f"\nEstimated State Error Statistics:")
    print(f"  Mean: {np.mean(est_error):.3f} m")
    print(f"  Std:  {np.std(est_error):.3f} m")
    print(f"  Max:  {np.max(est_error):.3f} m")
    
    # Coefficient of improvement
    improvement = (np.mean(gnss_error) - np.mean(est_error)) / np.mean(gnss_error) * 100
    print(f"\nEstimation Improvement: {improvement:.1f}%")
    
    # Best practices
    print("\n6. Best Practices")
    print("-" * 40)
    
    practices = [
        "1. Estimate your GPS accuracy BEFORE using it in the filter",
        "2. Start with conservative (large) noise covariance",
        "3. Monitor innovation (z - h(x)) to detect GPS outliers",
        "4. Keep landmarks even with GNSS for robustness",
        "5. Test your system with and without GNSS",
        "6. Log innovation and covariance for analysis",
        "7. Use adaptive tuning for time-varying conditions",
    ]
    
    for practice in practices:
        print(f"  • {practice}")
    
    print("\n" + "=" * 60)
    print("End of Example")
    print("=" * 60)


if __name__ == "__main__":
    example_gnss_slam()
