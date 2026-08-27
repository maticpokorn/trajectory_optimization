import numpy as np
from tqdm import tqdm
from source.solver import NoTimestampSolver
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from time import time


def run_convergence_time_analysis(n, n_iter, instance):
    travel_time = n
    solver = NoTimestampSolver(d=7, r=4, q=4, dimensions=3)

    # Store results as lists of (time, improvement) pairs for each run
    results_newton = []
    results_newton2 = []
    results_grad = []
    results_scipy = []
    results_tc = []

    for i in tqdm(range(n_iter)):
        np.random.seed(i)
        if instance == "random":
            waypoints = np.random.rand(n, 3)
        elif instance == "random_and_line":
            waypoints = (
                np.random.rand(n, 3)
                + np.array([np.arange(n), np.zeros(n), np.zeros(n)]).T
            )
        dists = np.linalg.norm((waypoints[1:] - waypoints[:-1]), axis=1)
        t = travel_time * dists / np.sum(dists)
        start_obj = solver.timestamp_solver.solve_sparse(waypoints, t)

        # NEWTON
        start_time = time()
        obj_newton, intermediate_results_newton, _, intermediate_times_newton = (
            solver.solve_backtracking_line_search(
                waypoints, travel_time, tau=0.5, c=0.5, bands=4
            )
        )
        total_time_newton = time() - start_time

        if np.isnan(obj_newton):
            print("NaN encountered in Newton's method. Breaking.")
            break

        # Use actual timestamps from the solver
        improvements_newton = (
            start_obj - np.array(intermediate_results_newton)
        ) / start_obj
        results_newton.append(list(zip(intermediate_times_newton, improvements_newton)))

        # NEWTON2
        start_time = time()
        obj_newton2, intermediate_results_newton2, _, intermediate_times_newton2 = (
            solver.solve_backtracking_line_search2(
                waypoints, travel_time, tau=0.5, c=0.5, bands=4
            )
        )
        total_time_newton2 = time() - start_time

        improvements_newton2 = (
            start_obj - np.array(intermediate_results_newton2)
        ) / start_obj
        results_newton2.append(
            list(zip(intermediate_times_newton2, improvements_newton2))
        )

        # GRADIENT
        start_time = time()
        obj_grad, intermediate_results_grad, intermediate_times_grad = (
            solver.solve_backtracking_line_search_grad(
                waypoints, travel_time, tau=0.5, c=0.5
            )
        )
        total_time_grad = time() - start_time

        improvements_grad = (
            start_obj - np.array(intermediate_results_grad)
        ) / start_obj
        results_grad.append(list(zip(intermediate_times_grad, improvements_grad)))

        # SCIPY
        start_time = time()
        obj_scipy, intermediate_results_scipy, intermediate_times_scipy = (
            solver.solve_scipy(waypoints, travel_time, use_jac=False, use_hess=False)
        )
        total_time_scipy = time() - start_time

        improvements_scipy = (
            start_obj - np.array(intermediate_results_scipy)
        ) / start_obj
        results_scipy.append(list(zip(intermediate_times_scipy, improvements_scipy)))

        # TRUST-CONSTR
        start_time = time()
        obj_tc, intermediate_results_tc, intermediate_times_tc = (
            solver.solve_trust_constr(waypoints, travel_time)
        )
        total_time_tc = time() - start_time

        improvements_tc = (start_obj - np.array(intermediate_results_tc)) / start_obj
        results_tc.append(list(zip(intermediate_times_tc, improvements_tc)))

    # Find the maximum time across all methods to create a common time grid
    max_time = 0
    for results in [
        results_newton,
        results_newton2,
        results_grad,
        results_scipy,
        results_tc,
    ]:
        for run in results:
            if len(run) > 0:
                max_time = max(max_time, run[-1][0])

    # Create a common time grid
    time_grid = np.linspace(0, max_time, 200)

    # Interpolate each method's results onto the common time grid
    datasets = [
        (results_newton, "blue", "newton"),
        (results_newton2, "cyan", "newton2"),
        (results_grad, "red", "gradient"),
        (results_scipy, "orange", "scipy"),
        (results_tc, "purple", "trust-constr"),
    ]

    csv_data = {"time": time_grid}

    for results, color, label in datasets:
        interpolated_improvements = []

        for run in results:
            if len(run) == 0:
                continue
            times, improvements = zip(*run)
            times = np.array(times)
            improvements = np.array(improvements)

            # Interpolate onto the common time grid
            interp_improvements = np.interp(
                time_grid,
                times,
                improvements,
                left=improvements[0],
                right=improvements[-1],
            )
            interpolated_improvements.append(interp_improvements)

        if len(interpolated_improvements) > 0:
            interpolated_improvements = np.array(interpolated_improvements)
            mean_vals = interpolated_improvements.mean(axis=0)
            std_vals = interpolated_improvements.std(axis=0)

            csv_data[f"{label}_mean"] = mean_vals
            csv_data[f"{label}_std"] = std_vals

            plt.plot(time_grid, mean_vals, label=label, color=color)
            plt.fill_between(
                time_grid,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.2,
                edgecolor="none",
            )

    # Export to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(f"data/convergence_time_results_{instance}_{n}.csv", index=False)
    print(f"Results exported to data/convergence_time_results_{instance}_{n}.csv")

    """plt.xlabel("time (s)")
    plt.ylabel("average improvement")
    plt.legend()
    plt.title(f"Convergence analysis (wall clock time) - n={n}")
    plt.show()"""


if __name__ == "__main__":
    ns = [5, 10, 20, 30, 40]
    n_iter = 50
    instances = ["random_and_line"]
    for instance in instances:
        for n in ns:
            print(f"Running convergence time analysis for n={n}, instance={instance}")
            run_convergence_time_analysis(n, n_iter, instance)
