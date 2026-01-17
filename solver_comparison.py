from solver import NoTimestampSolver, TimestampSolver

# waypoints = [[0, 0.5], [-0.2, 0], [0, -0.5], [1, 0], [0.5, 0.5], [0.5, -1.5]]

# waypoints = [[0, 0], [0, 1], [0.5, 0.5], [1, 1], [1, 0]]

waypoints = [
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
]

t = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
T = 6

tsolver = TimestampSolver(7, 3, 3, 2)
tsolver.solve_sparse(waypoints, t)
tsolver.show_path()

ntsolver = NoTimestampSolver(7, 3, 3, 2)
ntsolver.solve_hybrid(waypoints, T, k=10)
ntsolver.show_path()
