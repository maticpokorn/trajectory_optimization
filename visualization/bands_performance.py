from source.solver import NoTimestampSolver
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path

DATA_DIR = Path("data")
IMG_DIR = Path("img")
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

n = 30
n_iter = 20
np.random.seed(0)

waypoints = np.random.rand(n, 3)

travel_time = n

# Range of bands to test
band_values = [None, 1, 2, 3, 4, 5, 7, 10]
methods = band_values + ["grad"]  # Add gradient method

converged_costs = np.zeros((n_iter, len(methods)))
computation_times = np.zeros((n_iter, len(methods)))
intermediate_results_all = {}  # Store intermediate results for each method

for i in range(n_iter):
    # Test band-based methods
    for j, nb in enumerate(band_values):
        print(f"Testing with b={nb} bands...")
        solver = NoTimestampSolver(d=9, r=4, q=4, dimensions=3)

        start_time = time.time()
        obj, intermediate_results, intermediate_trajs = (
            solver.solve_backtracking_line_search(
                waypoints,
                travel_time,
                tau=0.5,
                c=0.5,
                intermediate_traj=False,
                bands=nb,
            )
        )
        end_time = time.time()

        converged_costs[i, j] = obj
        computation_times[i, j] = end_time - start_time

        # Store intermediate results
        if nb not in intermediate_results_all:
            intermediate_results_all[nb] = []
        intermediate_results_all[nb].append(intermediate_results)

        print(f"  Cost: {obj:.6f}, Time: {end_time - start_time:.3f}s")

    # Test gradient method
    print(f"Testing with gradient method...")
    solver = NoTimestampSolver(d=9, r=4, q=4, dimensions=3)

    start_time = time.time()
    obj, intermediate_results = solver.solve_backtracking_line_search_grad(
        waypoints,
        travel_time,
        tau=0.5,
        c=0.0,
    )
    end_time = time.time()

    grad_idx = len(band_values)
    converged_costs[i, grad_idx] = obj
    computation_times[i, grad_idx] = end_time - start_time

    # Store intermediate results
    if "grad" not in intermediate_results_all:
        intermediate_results_all["grad"] = []
    intermediate_results_all["grad"].append(intermediate_results)

    print(f"  Cost: {obj:.6f}, Time: {end_time - start_time:.3f}s")

# Prepare and export convergence data to CSV
csv_data = {}
max_iterations = max(
    max(len(res) for res in intermediate_results_all[method]) for method in methods
)

for method in methods:
    # Compute mean and std of intermediate results across all runs
    padded_results = []
    for res in intermediate_results_all[method]:
        padded = np.pad(res, (0, max_iterations - len(res)), "edge")
        padded_results.append(padded)

    padded_results = np.array(padded_results)
    mean_vals = np.mean(padded_results, axis=0)
    std_vals = np.std(padded_results, axis=0)

    if method == "grad":
        method_label = "grad"
    elif method is None:
        method_label = "qfull"
    else:
        method_label = f"q{method}"
    csv_data[f"{method_label}_mean"] = mean_vals
    csv_data[f"{method_label}_std"] = std_vals

csv_data["iteration"] = np.arange(max_iterations)
df = pd.DataFrame(csv_data)

# Reorder columns to have iteration first
columns_order = ["iteration"]
for method in methods:
    if method == "grad":
        method_label = "grad"
    elif method is None:
        method_label = "qfull"
    else:
        method_label = f"q{method}"
    columns_order.extend([f"{method_label}_mean", f"{method_label}_std"])
df = df[columns_order]

output_filename = DATA_DIR / f"bands_convergence_n{n}.csv"
df.to_csv(output_filename, index=False)
print(f"\nConvergence results exported to {output_filename}")

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot convergence curves for selected methods (left plot)
methods_to_plot = [
    None,
    1,
    2,
    3,
    4,
    5,
    7,
    10,
    "grad",
]  # Include None (full Hessian) and gradient method
colors = plt.cm.viridis(np.linspace(0, 1, len(methods_to_plot)))
markers = ["o", "s", "^", "D", "v", "p", "*", "h", "x", "+"]
linestyles = ["-", "--", "-.", ":", "-", "--"]

for idx, method in enumerate(methods_to_plot):
    if method in intermediate_results_all:
        # Average the intermediate results across the n_iter runs
        max_len = max(len(res) for res in intermediate_results_all[method])
        avg_convergence = np.zeros(max_len)

        for res in intermediate_results_all[method]:
            padded = np.pad(res, (0, max_len - len(res)), "edge")
            avg_convergence += padded

        avg_convergence /= len(intermediate_results_all[method])

        if method == "grad":
            label = "Gradient"
        elif method is None:
            label = "Full Hessian"
        else:
            label = f"q={method}"
        ax1.plot(
            avg_convergence,
            label=label,
            linewidth=2,
            color=colors[idx],
            marker=markers[idx % len(markers)],
            markersize=6,
            markevery=max(1, len(avg_convergence) // 10),  # Show ~10 markers
            linestyle=linestyles[idx % len(linestyles)],
        )

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Objective Value")
ax1.set_title("Convergence Comparison by Method")
ax1.legend()
ax1.grid(True)

# Plot computation time comparison (second plot) - bar chart
total_computation_times = []
method_labels_time = []

for idx, method in enumerate(methods):
    avg_time = np.mean(computation_times[:, idx])
    total_computation_times.append(avg_time)
    if method == "grad":
        label = "Gradient"
    elif method is None:
        label = "Full Hessian"
    else:
        label = f"q={method}"
    method_labels_time.append(label)

# Create bar plot for total computation time
x_pos_time = np.arange(len(method_labels_time))
colors_time = plt.cm.viridis(np.linspace(0, 1, len(method_labels_time)))
bars2 = ax2.bar(x_pos_time, total_computation_times, color=colors_time)
ax2.set_xlabel("Method")
ax2.set_ylabel("Computation Time (s)")
ax2.set_title("Total Computation Time")
ax2.set_xticks(x_pos_time)
ax2.set_xticklabels(method_labels_time, rotation=45, ha="right")
ax2.grid(True, axis="y")

# Plot computation time per iteration (third plot)
time_per_iteration = []
method_labels_plot = []

for idx, method in enumerate(methods):
    if method in intermediate_results_all:
        # Calculate average number of iterations
        avg_iterations = np.mean([len(res) for res in intermediate_results_all[method]])
        # Calculate average time per iteration
        avg_time = np.mean(computation_times[:, idx])
        avg_time_per_iter = avg_time / avg_iterations if avg_iterations > 0 else 0

        time_per_iteration.append(avg_time_per_iter)
        if method == "grad":
            label = "Gradient"
        elif method is None:
            label = "Full Hessian"
        else:
            label = f"q={method}"
        method_labels_plot.append(label)

# Create bar plot
x_pos = np.arange(len(method_labels_plot))
bars = ax3.bar(
    x_pos,
    time_per_iteration,
    color=(
        colors[: len(method_labels_plot)]
        if len(method_labels_plot) <= len(colors)
        else plt.cm.viridis(np.linspace(0, 1, len(method_labels_plot)))
    ),
)
ax3.set_xlabel("Method")
ax3.set_ylabel("Time per Iteration (s)")
ax3.set_title("Average Time per Iteration")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(method_labels_plot, rotation=45, ha="right")
ax3.grid(True, axis="y")

plt.tight_layout()
plt.savefig(IMG_DIR / "bands_performance.pdf", bbox_inches="tight")
plt.show()

# Print summary
print("\n--- Summary ---")
best_idx = np.argmin(np.mean(converged_costs, axis=0))
best_method = methods[best_idx]
if best_method == "grad":
    best_method_label = "Gradient"
elif best_method is None:
    best_method_label = "Full Hessian"
else:
    best_method_label = f"q={best_method}"
print(
    f"Best cost: {np.min(np.mean(converged_costs, axis=0)):.6f} with {best_method_label}"
)

fastest_idx = np.argmin(np.mean(computation_times, axis=0))
fastest_method = methods[fastest_idx]
if fastest_method == "grad":
    fastest_method_label = "Gradient"
elif fastest_method is None:
    fastest_method_label = "Full Hessian"
else:
    fastest_method_label = f"q={fastest_method}"
print(
    f"Fastest time: {np.min(np.mean(computation_times, axis=0)):.3f}s with {fastest_method_label}"
)

# Export summary statistics to CSV
method_labels = []
for m in methods:
    if m == "grad":
        method_labels.append("Gradient")
    elif m is None:
        method_labels.append("Full Hessian")
    else:
        method_labels.append(f"q={m}")
summary_data = {
    "method": method_labels,
    "mean_cost": np.mean(converged_costs, axis=0),
    "std_cost": np.std(converged_costs, axis=0),
    "mean_time": np.mean(computation_times, axis=0),
    "std_time": np.std(computation_times, axis=0),
}
summary_df = pd.DataFrame(summary_data)
summary_filename = DATA_DIR / f"bands_summary_n{n}.csv"
summary_df.to_csv(summary_filename, index=False)
print(f"Summary statistics exported to {summary_filename}")

# Export per iteration time statistics to CSV
per_iter_data = {"method": method_labels}
for i in range(n_iter):
    run_per_iter_times = []
    for idx, method in enumerate(methods):
        if method in intermediate_results_all:
            num_iterations = len(intermediate_results_all[method][i])
            time_per_iter = (
                computation_times[i, idx] / num_iterations if num_iterations > 0 else 0
            )
            run_per_iter_times.append(time_per_iter)
        else:
            run_per_iter_times.append(0)
    per_iter_data[f"run_{i+1}"] = run_per_iter_times

# Add mean and std columns
per_iter_times_array = np.array([per_iter_data[f"run_{i+1}"] for i in range(n_iter)]).T
per_iter_data["mean"] = np.mean(per_iter_times_array, axis=1)
per_iter_data["std"] = np.std(per_iter_times_array, axis=1)

per_iter_df = pd.DataFrame(per_iter_data)
per_iter_filename = DATA_DIR / f"bands_per_iter_time_n{n}.csv"
per_iter_df.to_csv(per_iter_filename, index=False)
print(f"Per iteration time statistics exported to {per_iter_filename}")

# Export cumulative time statistics to CSV
cumulative_data = {"method": method_labels}
for i in range(n_iter):
    cumulative_data[f"run_{i+1}"] = computation_times[i, :]

# Add mean and std columns
cumulative_data["mean"] = np.mean(computation_times, axis=0)
cumulative_data["std"] = np.std(computation_times, axis=0)

cumulative_df = pd.DataFrame(cumulative_data)
cumulative_filename = DATA_DIR / f"bands_cumulative_time_n{n}.csv"
cumulative_df.to_csv(cumulative_filename, index=False)
print(f"Cumulative time statistics exported to {cumulative_filename}")
