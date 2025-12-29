from solver import NoTimestampSolver
import numpy as np
from time import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    waypoints = [
        [1, 0, 0],
        [2, -1, 1],
        [2, 0, 2],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1.5],
        [1, 1, 3],
        [2, 2, 2],
        [0, 3, 2],
        [0, 1, 2],
    ]
    n = 4
    np.random.seed(3)
    waypoints = (
        np.random.rand(n, 3) + np.array([np.arange(n), np.zeros(n), np.zeros(n)]).T
    )
    travel_time = n
    """
    create trajectory of polynomial splines of degree 7
    minimize norm of 3rd derivative (acceleration)
    enforce continuity up to 4th derivative
    trajectory in 3 dimesions
    travel time is 11s
    """

    solver = NoTimestampSolver(d=7, r=3, q=3, dimensions=1)
    """obj_hess, intermediate_results_hess = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=True
    )"""
    # obj_hess, intermediate_results_hess = solver.solve_newton(waypoints, travel_time)
    """start_time = time()
    obj_jac, intermediate_results_jac = solver.solve(waypoints, travel_time)
    print(f"time: {time() - start_time}")
    solver.show_path_3d(frame=0)"""
    """obj_jac, intermediate_results_jac = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=False
    )"""

    """obj, intermediate_results = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=True
    )"""
    start_time = time()
    obj_hybrid, intermediate_results_hybrid = solver.solve_hybrid(
        waypoints, travel_time, k=2, banded_hessian=False
    )
    print(f"time: {time() - start_time}")
    """plt.plot(
        range(len(intermediate_results_hess)),
        intermediate_results_hess,
        label="newton method",
    )"""
    plt.plot(
        range(len(intermediate_results_jac)),
        intermediate_results_jac,
        label="gradient descent",
        color="blue",
        marker="x",
    )
    """plt.plot(
        range(len(intermediate_results)),
        intermediate_results,
        label="trust-constr: hess, jac",
        color="lime",
        marker="x",
    )"""
    plt.plot(
        range(len(intermediate_results_hybrid)),
        intermediate_results_hybrid,
        label="hybrid",
        color="red",
        marker="x",
    )
    # plt.ylim([0.98 * min(intermediate_results_hybrid), 1.2 * intermediate_results[1]])
    plt.legend()
    plt.yscale("log")
    plt.show()
    # solver.show_path()
