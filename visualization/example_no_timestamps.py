from source.solver import NoTimestampSolver
import numpy as np
from time import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """waypoints = [
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
    n = len(waypoints)
    travel_time = 10"""
    n = 20
    np.random.seed(0)
    waypoints = np.random.rand(
        n, 3
    )  # + np.array([np.arange(n), np.arange(n), np.arange(n)]).T
    travel_time = n
    """
    create trajectory of polynomial splines of degree 7
    minimize norm of 3rd derivative (acceleration)
    enforce continuity up to 4th derivative
    trajectory in 3 dimesions
    travel time is 11s
    """

    solver = NoTimestampSolver(d=7, r=3, q=3, dimensions=3)
    """obj_hess, intermediate_results_hess = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=True
    )"""
    # obj_hess, intermediate_results_hess = solver.solve_newton(waypoints, travel_time)
    start_time = time()
    obj_jac, intermediate_results_jac = solver.solve(waypoints, travel_time)
    time_gradient = time() - start_time
    print(f"time gradient descent: {time_gradient}")
    # solver.show_path_3d(frame=0)
    """obj_jac, intermediate_results_jac = solver.solve_scipy(
        waypoints, travel_time, use_jac=True, use_hess=False
    )"""

    start_time = time()
    obj, intermediate_results, _ = solver.solve_trust_constr(waypoints, travel_time)
    time_scipy = time() - start_time
    print(f"time scipy: {time_scipy}")
    # solver.show_path_3d(frame=0)

    """start_time = time()
    obj_hybrid, intermediate_results_hybrid = solver.solve_hybrid(
        waypoints, travel_time, k=5, banded_hessian=False
    )
    print(f"time: {time() - start_time}")"""
    print("started backtracking line search")
    start_time = time()
    obj_line, intermediate_results_line, _, _ = solver.solve_backtracking_line_search(
        waypoints, travel_time, tau=0.5, c=0.5
    )
    time_backtrack_newton = time() - start_time
    print(f"time backtracking newton: {time_backtrack_newton}")

    start_time = time()
    obj_line_grad, intermediate_results_line_grad, _ = (
        solver.solve_backtracking_line_search_grad(
            waypoints, travel_time, tau=0.5, c=0.5
        )
    )
    time_backtrack_grad = time() - start_time
    print(f"time backtracking grad: {time_backtrack_grad}")

    start_time = time()
    obj_line_hess, intermediate_results_line_hess, _, _ = (
        solver.solve_backtracking_line_search_hess(
            waypoints, travel_time, tau=0.5, c=0.5
        )
    )
    time_backtrack_hess = time() - start_time
    print(f"time backtracking hess: {time_backtrack_hess}")

    start_time = time()
    obj_line_newton2, intermediate_results_line_newton2, _, _ = (
        solver.solve_backtracking_line_search2(waypoints, travel_time, tau=0.5, c=0.5)
    )
    time_backtrack_newton2 = time() - start_time
    print(f"time backtracking newton2: {time_backtrack_newton2}")

    start_time = time()
    obj_tc, intermediate_results_tc, _ = solver.solve_trust_constr(
        waypoints, travel_time
    )
    time_trust_constr = time() - start_time
    print(f"time trust-constr: {time_trust_constr}")
    # Print execution time summary
    print("\n" + "=" * 60)
    print("EXECUTION TIME SUMMARY")
    print("=" * 60)
    print(f"Gradient Descent:                    {time_gradient:.4f}s")
    print(f"Scipy (Hess + Jac):                  {time_scipy:.4f}s")
    print(f"Backtracking Line Search (Newton):   {time_backtrack_newton:.4f}s")
    print(f"Backtracking Line Search (Gradient): {time_backtrack_grad:.4f}s")
    print(f"Backtracking Line Search (Hessian):  {time_backtrack_hess:.4f}s")
    print(f"Backtracking Line Search (Newton2):  {time_backtrack_newton2:.4f}s")
    print(f"Trust-Constr:                        {time_trust_constr:.4f}s")
    print("=" * 60 + "\n")

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
    plt.plot(
        range(len(intermediate_results)),
        intermediate_results,
        label="scipy: hess, jac",
        color="orange",
        marker="o",
    )
    """plt.plot(
        range(len(intermediate_results_hybrid)),
        intermediate_results_hybrid,
        label="newton (hybrid)",
        color="red",
        marker="x",
    )"""
    plt.plot(
        range(len(intermediate_results_line)),
        intermediate_results_line,
        label="newton (backtracking line search)",
        color="lime",
        marker="x",
    )
    plt.plot(
        range(len(intermediate_results_line_grad)),
        intermediate_results_line_grad,
        label="gradient (backtracking line search)",
        color="red",
        marker="x",
    )
    plt.plot(
        range(len(intermediate_results_line_hess)),
        intermediate_results_line_hess,
        label="hessian (backtracking line search)",
        color="magenta",
        marker="s",
    )
    plt.plot(
        range(len(intermediate_results_line_newton2)),
        intermediate_results_line_newton2,
        label="newton2 (backtracking line search)",
        color="cyan",
        marker="d",
    )
    plt.plot(
        range(len(intermediate_results_tc)),
        intermediate_results_tc,
        label="trust-constr",
        color="purple",
        marker="x",
    )
    # plt.ylim([0.98 * min(intermediate_results_hybrid), 1.2 * intermediate_results[1]])
    plt.legend()
    # plt.yscale("log")
    plt.title(f"num. waypoints: {n}")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()
    # solver.show_path()
