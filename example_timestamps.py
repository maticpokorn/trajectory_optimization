from solver import TimestampSolver

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
    ]
    t = [1] * (len(waypoints) - 1)
    """
    create trajectory of polynomial splines of degree 7
    minimize norm of 3rd derivative (acceleration)
    enforce continuity up to 4th derivative
    trajectory in 3 dimesions
    travel time is 1s per segment
    """
    solver = TimestampSolver(d=7, r=3, q=4, dimensions=3)
    solver.solve(waypoints, t)
    print(f"optimal objective: {solver.obj}")
    solver.show_path()
