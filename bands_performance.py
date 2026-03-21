from solver import NoTimestampSolver
import numpy as np
import matplotlib.pyplot as plt

# Generate 40 waypoints along a complex path
np.random.seed(42)
t_path = np.linspace(0, 4 * np.pi, 20)
waypoints = []
for t in t_path:
    x = 10 * np.cos(t) + 2 * np.cos(3 * t)
    y = 10 * np.sin(t) + 2 * np.sin(3 * t)
    waypoints.append([x, y, 0])

travel_time = 20

# Range of bands to test
band_values = range(1, 11)  # Test q from 1 to 10

converged_costs = []
computation_times = []

for nb in band_values:
    print(f"Testing with b={nb} bands...")
    solver = NoTimestampSolver(d=9, r=4, q=4, dimensions=3)

    import time

    start_time = time.time()
    obj, intermediate_results, intermediate_trajs = (
        solver.solve_backtracking_line_search(
            waypoints, travel_time, tau=0.5, c=0.5, intermediate_traj=False, bands=nb
        )
    )
    end_time = time.time()

    converged_costs.append(obj)
    computation_times.append(end_time - start_time)
    print(f"  Cost: {obj:.6f}, Time: {end_time - start_time:.3f}s")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot converged cost vs number of bands
ax1.plot(band_values, converged_costs, "o-", linewidth=2, markersize=8)
ax1.set_xlabel("Number of Bands (q)")
ax1.set_ylabel("Converged Cost Function Value")
ax1.set_title("Cost Function vs Number of Bands")
ax1.grid(True)

# Plot computation time vs number of bands
ax2.plot(
    band_values, computation_times, "s-", linewidth=2, markersize=8, color="orange"
)
ax2.set_xlabel("Number of Bands (q)")
ax2.set_ylabel("Computation Time (s)")
ax2.set_title("Computation Time vs Number of Bands")
ax2.grid(True)

plt.tight_layout()
plt.savefig("bands_performance.pdf", bbox_inches="tight")
plt.show()

# Print summary
print("\n--- Summary ---")
print(
    f"Best cost: {min(converged_costs):.6f} at q={band_values[np.argmin(converged_costs)]}"
)
print(
    f"Fastest time: {min(computation_times):.3f}s at q={band_values[np.argmin(computation_times)]}"
)
