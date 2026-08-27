from source.solver import NoTimestampSolver

waypoints = [[0, 0, 0], [1, 0, 1], [1, -1, 1], [1, 1, 0], [0, -1, 0], [1, 0, 0]]

T = 6

solver = NoTimestampSolver(d=7, r=3, q=3, dimensions=3)
solver.solve_backtracking_line_search(waypoints, T)
solver.show_path_3d_latex(frame=0)
