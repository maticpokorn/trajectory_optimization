from solver import TimestampSolver, NoTimestampSolver
import numpy as np

waypoints = np.array([[1, 0], [1, 1], [2, 0]])
t = [1.0, 2.0]

# solver = TimestampSolver(d=4, r=3, q=2, dimensions=2)
solver = NoTimestampSolver(d=4, r=3, q=2, dimensions=2)
# solver.solve_sparse(waypoints, t)
solver.solve_backtracking_line_search(waypoints, sum(t), x0=t)
# solver.show_path_colored_2d()
solver.timestamp_solver.show_path_colored_2d()

# print(solver.grad(waypoints, t))
