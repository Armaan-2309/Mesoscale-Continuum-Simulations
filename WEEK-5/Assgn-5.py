import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Parameters
L = 5.0                 
num_particles = 5
t_total = 100
dt = 0.001
N = int(t_total / dt)

# Simulate Brownian motion
def simulate_particles_with_wrapping(num_particles=5, L=5.0, t_total=100, dt=0.001):
    steps = int(t_total / dt)
    unwrapped_pos = np.zeros((steps + 1, num_particles, 3))
    wrapped_pos = np.zeros_like(unwrapped_pos)

    # Random non-overlapping initial positions
    initial_positions = []
    min_dist = 0.5
    while len(initial_positions) < num_particles:
        candidate = np.random.uniform(0, L, 3)
        if all(np.linalg.norm(candidate - p) > min_dist for p in initial_positions):
            initial_positions.append(candidate)
    unwrapped_pos[0] = np.array(initial_positions)
    wrapped_pos[0] = unwrapped_pos[0] % L

    # Time evolution
    for i in range(1, steps + 1):
        n = np.random.uniform(-1, 1, (num_particles, 3))
        delta_r = 6 * np.sqrt(dt) * n
        unwrapped_pos[i] = unwrapped_pos[i - 1] + delta_r
        wrapped_pos[i] = unwrapped_pos[i] % L

    return unwrapped_pos, wrapped_pos

# Compute MSD
def compute_and_plot_msd(pos_all, dt, t_max=10):
    steps = int(t_max / dt)
    N_particles = pos_all.shape[1]
    MSD_individual = np.zeros((N_particles, steps))

    for tau in range(1, steps + 1):
        for p in range(N_particles):
            disp = pos_all[tau:, p, :] - pos_all[:-tau, p, :]
            sq_disp = np.sum(disp ** 2, axis=1)
            MSD_individual[p, tau - 1] = np.mean(sq_disp)

    avg_msd = np.mean(MSD_individual, axis=0)
    t_msd = np.arange(1, steps + 1) * dt

    # Linear fit to get D*
    slope, intercept, *_ = linregress(t_msd[:1000], avg_msd[:1000])
    D_star = slope / 6
    print(f"Estimated diffusivity D* = {D_star:.4f}")

    # Plot MSDs
    plt.figure(figsize=(8, 6))
    for i in range(N_particles):
        plt.plot(t_msd, MSD_individual[i], label=f'Particle {i+1}', alpha=0.6)
    plt.plot(t_msd, avg_msd, 'k-', linewidth=2.5, label='Average MSD')
    plt.xlabel("t*")
    plt.ylabel("MSD")
    plt.title("Individual and Average MSD vs t*")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("individual_and_average_msd.png")
    plt.show()

    return t_msd, avg_msd, D_star


unwrapped_positions, wrapped_positions = simulate_particles_with_wrapping(
    num_particles=num_particles, L=L, t_total=t_total, dt=dt
)

# Save wrapped trajectories
time_vec = np.arange(0, t_total + dt, dt)
for p in range(num_particles):
    np.savetxt(f"trajectory_particle_{p+1}.txt",
               np.column_stack((time_vec, wrapped_positions[:, p, :])),
               header="t* x* y* z*")

# Compute and plot MSD
t_msd, msd_vals, D_star = compute_and_plot_msd(unwrapped_positions, dt, t_max=10)
