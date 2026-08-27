from source.solver import NoTimestampSolver
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib to use LaTeX fonts
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

waypoints = [
    [0, 0, 0],
    [8, 4, 0],
    [2, 10, 0],
    [12, 14, 0],
    [8, 22, 0],
    [3, 20, 0],
    [0, 25, 0],
    [-8, 20, 0],
    [-4, 12, 0],
    [0, 8, 0],
    [4, 4, 0],
    [0, 0, 0],
]
travel_time = 10
solver = NoTimestampSolver(d=9, r=4, q=4, dimensions=3)
obj, intermediate_results, intermediate_trajs, intermediate_times = (
    solver.solve_backtracking_line_search(
        waypoints, travel_time, tau=0.5, c=0.5, intermediate_traj=True
    )
)

t = np.linspace(0, travel_time, 1000)
fig = plt.figure(figsize=(9, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.25, wspace=0.2)
axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
colors = plt.cm.plasma(np.linspace(0, 1, len(intermediate_trajs)))
alphas = np.linspace(0.3, 1.0, len(intermediate_trajs))

# Add waypoints
waypoints_array = np.array(waypoints)

for i, traj in enumerate(intermediate_trajs):
    pos = np.array([traj.evaluate(ti) for ti in t])
    waypoint_times = np.cumsum([0] + list(traj.t))

    axs[0].plot(
        pos[:, 0], pos[:, 1], color=colors[i], alpha=alphas[i], label=f"Iter: {i+1}"
    )
    axs[0].scatter(
        waypoints_array[:, 0],
        waypoints_array[:, 1],
        color=colors[i],
        s=30,
        zorder=5,
        alpha=alphas[i],
        edgecolors="black",
        linewidths=0.5,
    )

    axs[1].plot(t, pos[:, 0], color=colors[i], alpha=alphas[i], label=f"Iter: {i+1}")
    axs[1].scatter(
        waypoint_times,
        waypoints_array[:, 0],
        color=colors[i],
        s=30,
        zorder=5,
        alpha=alphas[i],
        edgecolors="black",
        linewidths=0.5,
    )

    axs[2].plot(t, pos[:, 1], color=colors[i], alpha=alphas[i])
    axs[2].scatter(
        waypoint_times,
        waypoints_array[:, 1],
        color=colors[i],
        s=30,
        zorder=5,
        alpha=alphas[i],
        edgecolors="black",
        linewidths=0.5,
    )

axs[0].set_ylabel("y")
axs[0].set_xlabel("x")
axs[0].set_title("Trajektorija v ravnini z=0")
axs[0].grid(True)
axs[0].axis("equal")
axs[0].legend(loc="lower right")

axs[1].set_ylabel("x")
axs[1].set_xlabel("Čas (s)")
axs[1].set_title("Koordinata x")
axs[1].grid(True)

axs[2].set_ylabel("y")
axs[2].set_xlabel("Čas (s)")
axs[2].set_title("Koordinata y")
axs[2].grid(True)

# fig.suptitle("Trajectory Optimization Progress", fontsize=16)
plt.tight_layout()
plt.savefig("img/trajectory_optimization.pdf", bbox_inches="tight")
plt.show()
