import numpy as np
from tqdm import tqdm
from solver import NoTimestampSolver
import matplotlib.pyplot as plt
import pandas as pd

n = 50
travel_time = n
solver = NoTimestampSolver(d=9, r=4, q=4, dimensions=3)

n_iter = 20
sum_improvements_newton = np.zeros((n_iter, 50))
sum_improvements_grad = np.zeros((n_iter, 50))
# sum_improvements = np.zeros((n_iter, 50))
sum_improvements_tc = np.zeros((n_iter, 50))


for i in tqdm(range(n_iter)):
    np.random.seed(i)
    waypoints = (
        np.random.rand(n, 3) + np.array([np.arange(n), np.zeros(n), np.zeros(n)]).T
    )
    dists = np.linalg.norm((waypoints[1:] - waypoints[:-1]), axis=1)
    t = travel_time * dists / np.sum(dists)
    start_obj = solver.timestamp_solver.solve_sparse(waypoints, t)

    # NEWTON
    obj_newton, intermediate_results_newton, _ = solver.solve_backtracking_line_search(
        waypoints, travel_time, tau=0.5, c=0.5
    )
    sum_improvements_newton[i, : len(intermediate_results_newton)] += (
        start_obj - np.array(intermediate_results_newton)
    ) / start_obj
    sum_improvements_newton[i, len(intermediate_results_newton) :] += (
        start_obj - obj_newton
    ) / start_obj

    # GRADIENT
    obj_grad, intermediate_results_grad = solver.solve_backtracking_line_search_grad(
        waypoints, travel_time, tau=0.5, c=0.5
    )
    sum_improvements_grad[i, : len(intermediate_results_grad)] += (
        start_obj - np.array(intermediate_results_grad)
    ) / start_obj
    sum_improvements_grad[i, len(intermediate_results_grad) :] += (
        start_obj - obj_grad
    ) / start_obj

    # NOTHING
    obj_tc, intermediate_results_tc = solver.solve_scipy(
        waypoints, travel_time, use_jac=False, use_hess=False
    )
    sum_improvements_tc[i, : len(intermediate_results_tc)] += (
        start_obj - np.array(intermediate_results_tc)
    ) / start_obj
    sum_improvements_tc[i, len(intermediate_results_tc) :] += (
        start_obj - obj_tc
    ) / start_obj
    """
    obj_tc, intermediate_results_tc = solver.solve_scipy(
        waypoints, travel_time, use_jac=False, use_hess=False
    )
    sum_improvements_tc[i, : len(intermediate_results_tc)] += (
        start_obj - np.array(intermediate_results_tc)
    ) / start_obj
    sum_improvements_tc[i, len(intermediate_results_tc) :] += (
        start_obj - obj_tc
    ) / start_obj"""

datasets = [
    (sum_improvements_newton, "blue", "newton"),
    (sum_improvements_grad, "red", "gradient"),
    (sum_improvements_tc, "limegreen", "trust-constr"),
]

# Prepare data for CSV export
csv_data = {}
for dataset in datasets:
    sum_improvements, color, label = dataset
    mean_vals = sum_improvements.mean(axis=0)[1:]
    std_vals = sum_improvements.std(axis=0)[1:]

    # Store data for CSV
    csv_data[f"{label}_mean"] = mean_vals
    csv_data[f"{label}_std"] = std_vals

    plt.plot(mean_vals, label=label, color=color)
    """plt.fill_between(
        np.arange(len(mean_vals)),
        mean_vals - std_vals,
        mean_vals + std_vals,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )"""

# Export to CSV
csv_data["iteration"] = np.arange(len(mean_vals))
df = pd.DataFrame(csv_data)
# Reorder columns to have iteration first
columns_order = [
    "iteration",
    "newton_mean",
    "newton_std",
    "gradient_mean",
    "gradient_std",
    "trust-constr_mean",
    "trust-constr_std",
]
df = df[columns_order]
df.to_csv("convergence_results.csv", index=False)
print(f"Results exported to convergence_results.csv")

plt.xlabel("iteration")
plt.ylabel("average improvement")
plt.legend()
plt.title("Convergence analysis")
plt.show()
