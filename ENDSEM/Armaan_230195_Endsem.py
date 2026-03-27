import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.setrecursionlimit(10000)

# Parameters
N_steps = 500
N_configs = 500
directions = [(1, 0), (1, 1), (-1, 0), (1, -1), (-1, 1), (-1, -1)]

ideal_distances = []
no_overlap_distances = []

# Part 1: Overlap allowed
for _ in range(N_configs):
    x, y = 0, 0
    for _ in range(N_steps):
        dx, dy = random.choice(directions)
        x += dx
        y += dy
    R = np.sqrt(x**2 + y**2)
    ideal_distances.append(R)

# Part 2: Overlap not allowed

def build_chain(x, y, visited, path, steps_remaining):
    if steps_remaining == 0:
        return True
    shuffled_dirs = directions[:]
    random.shuffle(shuffled_dirs)
    for dx, dy in shuffled_dirs:
        nx, ny = x + dx, y + dy
        if (nx, ny) not in visited:
            visited.add((nx, ny))
            path.append((nx, ny))
            if build_chain(nx, ny, visited, path, steps_remaining - 1):
                return True
            visited.remove((nx, ny))
            path.pop()
    return False

accepted = 0
attempts = 0

while accepted < N_configs:
    attempts += 1
    visited = set()
    path = [(0, 0)]
    visited.add((0, 0))
    success = build_chain(0, 0, visited, path, N_steps)
    if success:
        x0, y0 = path[0]
        xN, yN = path[-1]
        R = np.sqrt((xN - x0)**2 + (yN - y0)**2)
        no_overlap_distances.append(R)
        accepted += 1
    if attempts % 10 == 0:
        print(f"Accepted: {accepted} / {attempts}", end='\r')

print(f"\nAcceptance rate: {100 * accepted / attempts:.2f}%")

# Plotting
plt.figure(figsize=(9, 6))
plt.hist(ideal_distances, bins=100, density=True, alpha=0.7, label="Overlap Allowed", color="teal", histtype='step', linewidth=2)
plt.hist(no_overlap_distances, bins=100, density=True, alpha=0.7, label="Overlap Not Allowed (Backtracking)", color="tomato", linestyle="dashed", histtype='step', linewidth=2)
plt.xlabel("End-to-End Distance", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.title("End-to-End Distance Distribution\n(With vs Without Bead Overlaps)", fontsize=13)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

mean_ideal = np.mean(ideal_distances)
mean_no_overlap = np.mean(no_overlap_distances)

# Mode Calculation
def mode_from_hist(data):
    counts, bins = np.histogram(data, bins=100)
    max_index = np.argmax(counts)
    return (bins[max_index] + bins[max_index+1]) / 2

mode_ideal = mode_from_hist(ideal_distances)
mode_no_overlap = mode_from_hist(no_overlap_distances)

print(f"\nAverage End-to-End Distance (Overlap Allowed): {mean_ideal:.3f}")
print(f"Most Likely End-to-End Distance (Mode): {mode_ideal:.3f}")
print(f"Average End-to-End Distance (No Overlap): {mean_no_overlap:.3f}")
print(f"Most Likely End-to-End Distance (Mode): {mode_no_overlap:.3f}")
