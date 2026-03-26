import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Define function
def run_simulation(gamma_dot_star, v=500, delta_t_star=0.01, total_t_star=10000, output_interval=100):
    steps = int(total_t_star / delta_t_star)
    
    # Initial bead positions
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([np.sqrt(v), 0.0, 0.0])
    r3 = np.array([2 * np.sqrt(v), 0.0, 0.0])
    r4 = np.array([3 * np.sqrt(v), 0.0, 0.0])
    
    sigma_xy = []
    
    def spring_force(ri, rj):
        R = rj - ri
        R_mag = np.linalg.norm(R)
        r_hat = R_mag / v
        if abs(1 - r_hat**2) < 1e-6:
            r_hat = 0.999
        return ((3 - r_hat**2) / (v * (1 - r_hat**2))) * R, R

    for step in range(steps + 1):
        # Brownian forces
        B1 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
        B2 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
        B3 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
        B4 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)

        # Shear flow
        flow1 = np.array([gamma_dot_star * r1[1], 0.0, 0.0])
        flow2 = np.array([gamma_dot_star * r2[1], 0.0, 0.0])
        flow3 = np.array([gamma_dot_star * r3[1], 0.0, 0.0])
        flow4 = np.array([gamma_dot_star * r4[1], 0.0, 0.0])

        # Spring forces
        F12, R12 = spring_force(r1, r2)
        F23, R23 = spring_force(r2, r3)
        F34, R34 = spring_force(r3, r4)

        # Update positions
        r1 += delta_t_star * (flow1 + B1 + F12)
        r2 += delta_t_star * (flow2 + B2 - F12 + F23)
        r3 += delta_t_star * (flow3 + B3 - F23 + F34)
        r4 += delta_t_star * (flow4 + B4 - F34)

        # Record stress
        if step % output_interval == 0:
            stress = (
                F12[0] * R12[1] +
                F23[0] * R23[1] +
                F34[0] * R34[1]
            )
            sigma_xy.append(stress)

    # Average stress over last 20%
    sigma_xy = np.array(sigma_xy)
    start_idx = int(0.8 * len(sigma_xy))
    mean_stress = np.mean(sigma_xy[start_idx:])
    viscosity = mean_stress / gamma_dot_star

    return viscosity, mean_stress


# RUN FOR MULTIPLE γ̇* VALUES
gamma_dot_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
viscosity_list = []
stress_list = []

print("Running simulations for various shear rates...")
for gamma_dot in gamma_dot_list:
    print(f"--> gamma_dot* = {gamma_dot}")
    eta, sigma = run_simulation(gamma_dot)
    viscosity_list.append(eta)
    stress_list.append(sigma)
    print(f"   eta = {eta:.4f}, avg_sigma = {sigma:.4f}")


# Smooth viscosity values
smooth_eta = gaussian_filter1d(viscosity_list, sigma=1)

plt.figure(figsize=(7, 5))
plt.loglog(gamma_dot_list, viscosity_list, 'ko--', label='Raw η')
plt.loglog(gamma_dot_list, smooth_eta, 'r-', linewidth=2, label='Smoothed η')
plt.xlabel("Shear Rate γ̇*", fontsize=12)
plt.ylabel("Viscosity η", fontsize=12)
plt.title("Flow Curve: Viscosity vs Shear Rate", fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("flow_curve_smoothed.png")
plt.show()

