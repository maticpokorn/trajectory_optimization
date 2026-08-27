import numpy as np
from tqdm import tqdm
from source.solver import NoTimestampSolver
import matplotlib.pyplot as plt

bands = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = 100
travel_time = n
solver = NoTimestampSolver(d=9, r=4, q=4, dimensions=3)

n_iter = 1
improvements = np.zeros((len(bands), n_iter, 50))


for iter_idx in tqdm(range(n_iter)):
    np.random.seed(iter_idx)
    waypoints = (
        np.random.rand(n, 3) + np.array([np.arange(n), np.zeros(n), np.zeros(n)]).T
    )
    dists = np.linalg.norm((waypoints[1:] - waypoints[:-1]), axis=1)
    t = travel_time * dists / np.sum(dists)
    start_obj = solver.timestamp_solver.solve_sparse(waypoints, t)

    for i, band in enumerate(bands):
        # NEWTON
        obj_newton, intermediate_results_newton, _ = (
            solver.solve_backtracking_line_search(
                waypoints, travel_time, tau=0.5, c=0.5, bands=band
            )
        )
        improvements[i, iter_idx, : len(intermediate_results_newton)] = (
            start_obj - np.array(intermediate_results_newton)
        ) / start_obj
        improvements[i, iter_idx, len(intermediate_results_newton) :] = (
            start_obj - obj_newton
        ) / start_obj

for i, band in enumerate(bands):
    mean_vals = improvements[i].mean(axis=0)[1:]
    std_vals = improvements[i].std(axis=0)[1:]
    plt.plot(mean_vals, label=f"Bands: {band}")
    """plt.fill_between(
        np.arange(len(mean_vals)),
        mean_vals - std_vals,
        mean_vals + std_vals,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )"""
plt.xlabel("iteration")
plt.ylabel("average improvement")
plt.legend()
plt.title("Convergence analysis")
plt.show()
