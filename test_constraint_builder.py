from constraint_builder import LinearConstraintBuilder
from grad_tester import GradTester
import numpy as np
from time import time

t = [1, 0.9, 0.5]
x = [1, 0, 0.5, 1]

cb = LinearConstraintBuilder(x, 3, 2)

A, b, _ = cb.build(t)

from solver import TimestampSolver, NoTimestampSolver

if __name__ == "__main__":
    # solver = TimestampSolver(d=5, r=1, q=2, dimensions=3)
    waypoints = np.array(
        [
            [1, 0, 0],
            [2, -1, 1],
            [2, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 2, 2],
            [1, 1, 1],
            [1, 0, 0],
        ]
    )
    T = 10
    solver = NoTimestampSolver(d=7, r=4, q=3, dimensions=3)
    solver.solve(waypoints, T)
    solver.show_path()
    # solver.solve(waypoints, T)
    """dists = np.linalg.norm((waypoints[1:] - waypoints[:-1]), axis=1)
    # t = np.array([0.5, 1, 1.5, 1, 1.1])
    t = dists / np.sum(dists)

    solver.solve(waypoints, t)
    tester = GradTester(lambda t: solver.solve(waypoints, t), None)
    print(tester.numerical_grad(t))
    print(solver.grad(waypoints, t))"""

    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for sz in sizes:
        print(sz)

        waypoints = (
            np.random.rand(sz + 1, 3)
            + np.vstack(
                (np.arange(sz + 1), np.random.rand(sz + 1), np.random.rand(sz + 1))
            ).T
        )
        T = sz
        solver.solve(waypoints, T)
        solver.show_path()
