import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Configure matplotlib to use LaTeX fonts - sized for LaTeX figures
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

# Get n value from command line argument, default to 30
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30

print(f"Plotting results for n={n}")

# Load CSV files
convergence_df = pd.read_csv(f"bands_convergence_n{n}.csv")
summary_df = pd.read_csv(f"bands_summary_n{n}.csv")
per_iter_df = pd.read_csv(f"bands_per_iter_time_n{n}.csv")

# Extract method names and data
methods_to_plot = ["q1", "q2", "q3", "q4", "q5", "q7", "q10", "qfull"]
methods_for_left_plot = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q10",
    "qfull",
]  # Filtered methods for convergence plot
method_labels = {
    "qfull": "Vsi",
    "q1": "1",
    "q2": "2",
    "q3": "3",
    "q4": "4",
    "q5": "5",
    "q7": "7",
    "q10": "10",
}

# Create figure with three subplots - make bar charts narrower
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(10, 5), gridspec_kw={"width_ratios": [1.5, 0.65, 0.65]}
)

# First pass: determine best method based on total computation time (time to convergence)
temp_method_times = {}
for method in methods_to_plot:
    if method == "qfull":
        row = summary_df[summary_df["method"] == "Full Hessian"]
        if row.empty:
            row = summary_df[summary_df["method"] == "q=None"]
    else:
        method_name = method.replace("q", "q=")
        row = summary_df[summary_df["method"] == method_name]

    if not row.empty:
        temp_method_times[method] = row["mean_time"].values[0]

# Find the best (lowest time) method
if temp_method_times:
    best_method = min(temp_method_times.items(), key=lambda x: x[1])[0]
else:
    best_method = None

# Plot convergence curves (left plot) - showing average improvement
colors = plt.cm.plasma(np.linspace(0, 1, len(methods_for_left_plot)))
markers = ["o", "s", "^", "D", "v", "p", "*", "h", "x"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]

max_improvement = 0  # Track maximum improvement for ylim

for idx, method in enumerate(methods_for_left_plot):
    mean_col = f"{method}_mean"
    std_col = f"{method}_std"

    # Handle different naming conventions for full Hessian
    if method == "qfull":
        if mean_col not in convergence_df.columns:
            mean_col = "qNone_mean"
            std_col = "qNone_std"

    if mean_col in convergence_df.columns:
        iterations = convergence_df["iteration"].values
        mean_vals = convergence_df[mean_col].values

        # Calculate improvement: (initial - current) / initial
        initial_val = mean_vals[0]
        improvement = (initial_val - mean_vals) / initial_val

        # Track maximum improvement
        max_improvement = max(max_improvement, improvement.max())

        # Plot improvement curve
        alpha_val = 1.0 if method == best_method else 0.7
        zorder_val = 10 if method == best_method else 1
        ax1.plot(
            iterations,
            improvement,
            label=method_labels[method],
            linewidth=1,
            color=colors[idx],
            marker=markers[idx % len(markers)],
            markersize=6,
            markevery=1,  # Show marker at every iteration
            linestyle=linestyles[idx % len(linestyles)],
            alpha=alpha_val,
            zorder=zorder_val,
        )

ax1.set_xlabel("Iteracija")
ax1.set_ylabel("Povprečna izboljšava")
ax1.set_title("Primerjava konvergence po pasovih")
ax1.set_xlim([1, 7])
ax1.set_ylim([0.85, max_improvement * 1.01])  # Set ylim to max + 1%
ax1.legend()
ax1.grid(True)

# Plot computation time comparison (middle plot) - bar chart
method_labels_list = []
total_computation_times = []

for method in methods_to_plot:
    # Find matching row in summary_df
    if method == "qfull":
        # Try both possible labels for full Hessian
        row = summary_df[summary_df["method"] == "Full Hessian"]
        if row.empty:
            row = summary_df[summary_df["method"] == "q=None"]
    else:
        method_name = method.replace("q", "q=")
        row = summary_df[summary_df["method"] == method_name]

    if not row.empty:
        total_computation_times.append(row["mean_time"].values[0])
        method_labels_list.append(method_labels[method])

# Create bar plot for total computation time
x_pos_time = np.arange(len(method_labels_list))
colors_time = plt.cm.plasma(np.linspace(0, 1, len(method_labels_list)))
# Create bars individually with different alpha values
for i in range(len(method_labels_list)):
    alpha_val = 1.0 if methods_to_plot[i] == best_method else 0.7
    edgecolor = "black" if methods_to_plot[i] == best_method else "none"
    linewidth = 0.8 if methods_to_plot[i] == best_method else 0
    ax2.bar(
        x_pos_time[i],
        total_computation_times[i],
        color=colors_time[i],
        alpha=alpha_val,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
ax2.set_xlabel("Število pasov")
ax2.set_ylabel("Čas (s)")
ax2.set_title("Povp. čas do konvergence")
ax2.set_xticks(x_pos_time)
ax2.set_xticklabels(method_labels_list, rotation=45, ha="right")
ax2.grid(True, axis="y")

# Plot computation time per iteration (right plot)
time_per_iteration = []
method_labels_plot = []

for method in methods_to_plot:
    # Find matching row in per_iter_df
    if method == "qfull":
        # Try both possible labels for full Hessian
        row = per_iter_df[per_iter_df["method"] == "Full Hessian"]
        if row.empty:
            row = per_iter_df[per_iter_df["method"] == "q=None"]
    elif method == "grad":
        row = per_iter_df[per_iter_df["method"] == "Gradient"]
    else:
        method_name = method.replace("q", "q=")
        row = per_iter_df[per_iter_df["method"] == method_name]

    if not row.empty:
        time_per_iteration.append(row["mean"].values[0])
        method_labels_plot.append(method_labels[method])

# Create bar plot
x_pos = np.arange(len(method_labels_plot))
# Create bars individually with different alpha values
for i in range(len(method_labels_plot)):
    alpha_val = 1.0 if methods_to_plot[i] == best_method else 0.7
    edgecolor = "black" if methods_to_plot[i] == best_method else "none"
    linewidth = 0.8 if methods_to_plot[i] == best_method else 0
    ax3.bar(
        x_pos[i],
        time_per_iteration[i],
        color=plt.cm.plasma(np.linspace(0, 1, len(method_labels_plot)))[i],
        alpha=alpha_val,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
ax3.set_xlabel("Število pasov")
ax3.set_ylabel("Čas (s)")
ax3.set_title("Povp. čas na iteracijo")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(method_labels_plot, rotation=45, ha="right")
ax3.grid(True, axis="y")

plt.tight_layout()
output_filename = f"bands_performance_n{n}.pdf"
plt.savefig(output_filename, bbox_inches="tight")
print(f"Plot saved to {output_filename}")
plt.show()

# Print summary
best_cost_row = summary_df.loc[summary_df["mean_cost"].idxmin()]
print("\n--- Summary ---")
print(f"Best cost: {best_cost_row['mean_cost']:.6f} with {best_cost_row['method']}")

fastest_time_row = summary_df.loc[summary_df["mean_time"].idxmin()]
print(
    f"Fastest time: {fastest_time_row['mean_time']:.3f}s with {fastest_time_row['method']}"
)
