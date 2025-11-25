import numpy as np
import matplotlib.pyplot as plt

# Full matrix setup, 6 states
H0_full = np.diag([2, 4, 6, 6, 8, 10]) 
V_base_full = np.array([
    [2, 1, 1, 1, 1, 0],
    [1, 2, 1, 1, 0, 1],
    [1, 1, 2, 0, 1, 1],
    [1, 1, 0, 2, 1, 1],
    [1, 0, 1, 1, 2, 1],
    [0, 1, 1, 1, 1, 2]
])

# Reduced matrix (first 5 states)
H0_reduced = H0_full[:5, :5]
V_base_reduced = V_base_full[:5, :5]

g_values = np.linspace(-1, 1, 200)
ground_full = []
ground_reduced = []

for g in g_values:
    V_full = -0.5 * g * V_base_full
    H_full = H0_full + V_full
    eigvals_full = np.linalg.eigvalsh(H_full)
    ground_full.append(np.min(eigvals_full))

    V_reduced = -0.5 * g * V_base_reduced
    H_reduced = H0_reduced + V_reduced
    eigvals_reduced = np.linalg.eigvalsh(H_reduced)
    ground_reduced.append(np.min(eigvals_reduced))

ground_full = np.array(ground_full)
ground_reduced = np.array(ground_reduced)
energy_diff = ground_reduced - ground_full

plt.plot(g_values, energy_diff, label="Energy difference (approx - full)")
plt.xlabel("g")
plt.ylabel("Energy difference")
plt.title("Ground state energy difference vs g")
plt.legend()
plt.grid(True)
plt.show()
