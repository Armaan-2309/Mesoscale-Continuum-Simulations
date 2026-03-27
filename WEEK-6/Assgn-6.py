import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Parameters
N_total = 3000
box_size = 10.0
dt = 0.025
n_steps = 1000
rc = 1.0
a = 25.0
gamma = 4.5
sigma = 3.0
k_spring = 2.0
mass = 1.0
kT = 1.0

# Initialization of positions
def initialize_positions_grid(N, box_size):
    num_per_side = int(np.ceil(N ** (1 / 3)))
    spacing = box_size / num_per_side
    positions = []
    for x in range(num_per_side):
        for y in range(num_per_side):
            for z in range(num_per_side):
                if len(positions) >= N:
                    break
                jitter = np.random.uniform(-0.1, 0.1, 3)
                pos = np.array([x, y, z]) * spacing + jitter
                positions.append(pos)
    return np.array(positions[:N]) % box_size

# DPD force
def compute_dpd_forces(pos, vel, types):
    forces = np.zeros_like(pos)
    tree = cKDTree(pos, boxsize=box_size)
    pairs = tree.query_pairs(rc)

    for i, j in pairs:
        if types[i] == 'solvent' and types[j] == 'solvent':
            continue
        rij = pos[j] - pos[i]
        rij -= box_size * np.round(rij / box_size)
        dist = np.linalg.norm(rij)
        if dist > 1e-8:
            e_ij = rij / dist
            vij = vel[j] - vel[i]
            f_c = a * (1 - dist / rc) * e_ij
            wd = (1 - dist / rc) ** 2
            f_d = -gamma * wd * np.dot(vij, e_ij) * e_ij
            wr = (1 - dist / rc)
            theta = np.random.normal(0, 1)
            f_r = sigma * wr * theta / np.sqrt(dt) * e_ij
            f_total = f_c + f_d + f_r
            forces[i] += f_total
            forces[j] -= f_total
    return forces

# Spring force
def apply_spring_forces(forces, pos, bonds):
    for i, j in bonds:
        rij = pos[j] - pos[i]
        rij -= box_size * np.round(rij / box_size)
        f_spring = -k_spring * rij
        forces[i] += f_spring
        forces[j] -= f_spring
    return forces

# Velocity-Verlet integrator
def velocity_verlet(pos, vel, forces, bonds, types):
    vel += 0.5 * forces / mass * dt
    pos += vel * dt
    pos %= box_size
    new_forces = compute_dpd_forces(pos, vel, types)
    new_forces = apply_spring_forces(new_forces, pos, bonds)
    vel += 0.5 * new_forces / mass * dt
    return pos, vel, new_forces

# Main simulation
def run_simulation(N_polymer, num_chains=1):
    N_poly_total = N_polymer * num_chains
    N_solvent = N_total - N_poly_total
    types = np.array(['polymer'] * N_poly_total + ['solvent'] * N_solvent)
    positions = initialize_positions_grid(N_total, box_size)
    velocities = np.random.normal(0, np.sqrt(kT / mass), (N_total, 3))
    bonds = []
    for c in range(num_chains):
        offset = c * N_polymer
        for i in range(N_polymer - 1):
            bonds.append((offset + i, offset + i + 1))
    forces = compute_dpd_forces(positions, velocities, types)
    forces = apply_spring_forces(forces, positions, bonds)

    R_list = [[] for _ in range(num_chains)]

    for step in range(n_steps):
        positions, velocities, forces = velocity_verlet(positions, velocities, forces, bonds, types)
        if step % 10 == 0:
            for c in range(num_chains):
                start = c * N_polymer
                end = start + N_polymer - 1
                R = positions[end] - positions[start]
                R -= box_size * np.round(R / box_size)
                R_list[c].append(np.linalg.norm(R))

    return [np.array(R) for R in R_list]

# Run simulations
R_5a, R_5b = run_simulation(N_polymer=5, num_chains=2)
[R_10] = run_simulation(N_polymer=10, num_chains=1)

# Plot results
time = np.arange(len(R_10)) * dt * 10
plt.plot(time, R_5a, label='N=5 Chain 1')
plt.plot(time, R_5b, label='N=5 Chain 2')
plt.plot(time, R_10, label='N=10')
plt.xlabel("Time")
plt.ylabel("End-to-end distance |R|")
plt.title("DPD Polymer End-to-End Distance (3000 beads)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("end_to_end_comparison.png")
plt.show()

# Print results
print(f"Average <R> for N=5 Chain 1 = {np.mean(R_5a):.3f}")
print(f"Average <R> for N=5 Chain 2 = {np.mean(R_5b):.3f}")
print(f"Average <R> for N=10         = {np.mean(R_10):.3f}")
