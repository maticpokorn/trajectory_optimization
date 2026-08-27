from source.solver import NoTimestampSolver, TimestampSolver
import numpy as np

waypoints = [[0, 0.5], [-0.2, 0], [0, -0.5], [1, 0], [0.5, 0.5], [0.5, -1.5]]

# waypoints = [[0, 0], [0, 1], [0.5, 0.5], [1, 1], [1, 0]]

"""waypoints = [
    [-1, 1],
    [0, 1.1],
    [1, 1],
    [0, 1],
    [-0.2, 0.5],
    [0, 0],
    [1.5, 0],
    [2, 0.5],
    [2, 0],
    [3, 0.5],
    [3.5, 0.4],
    [3, 0.5],
    [3, 0],
    [3.2, 0],
    [3.5, 0.4],
    [3.5, 0],
    [5, 0],
]"""

t = np.ones(len(waypoints) - 1)
T = sum(t)

tsolver = TimestampSolver(9, 3, 3, 2)
tsolver.solve_sparse(waypoints, t)
tsolver.show_path_2d_latex(filename="img/qp_example.pdf")

ntsolver = NoTimestampSolver(9, 3, 3, 2)
ntsolver.solve_hybrid(waypoints, T, k=10)
ntsolver.show_path_2d_latex(filename="img/bilevel_example.pdf")
