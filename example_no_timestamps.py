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
        [1.5, 0, 1],
        [3, 0, 0],
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
        [1.5, 0, 1],
        [3, 0, 0],
    ]
    n = 50
    np.random.seed(11)
    waypoints = np.random.rand(n, 3)
    travel_time = n
    """
    create trajectory of polynomial splines of degree 7
    minimize norm of 3rd derivative (acceleration)
    enforce continuity up to 4th derivative
    trajectory in 3 dimesions
    travel time is 11s
    """
    start_time = time()
    solver = NoTimestampSolver(d=7, r=3, q=3, dimensions=3)
    obj_hess, intermediate_results_hess = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=True
    )

    obj_jac, intermediate_results_jac = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=False
    )

    obj, intermediate_results = solver.solve_scipy(
        waypoints, travel_time, use_jac=False, use_hess=False
    )
    plt.plot(
        range(len(intermediate_results_hess)),
        intermediate_results_hess,
        label="hess, jac",
    )
    plt.plot(
        range(len(intermediate_results_jac)),
        intermediate_results_jac,
        label="no hess, jac",
    )
    plt.plot(
        range(len(intermediate_results)), intermediate_results, label="no hess, no jac"
    )
    plt.legend()
    plt.show()
    # solver.show_path()
