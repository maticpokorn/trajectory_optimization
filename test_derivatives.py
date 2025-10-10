from trajectory import TimestampSolver
import numpy as np


def numerical_grad(f, x, eps=1e-6):
    """
    Compute gradient of scalar function f at x using central differences
    """
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_eps_plus = x.copy()
        x_eps_minus = x.copy()
        x_eps_plus[i] += eps
        x_eps_minus[i] -= eps
        grad[i] = (f(x_eps_plus) - f(x_eps_minus)) / (2 * eps)

    return grad


if __name__ == "__main__":
    solver = TimestampSolver(d=3, r=2, q=1, dimensions=2)
    waypoints = [
        [1, 0, 0],
        [2, -1, 1],
        [2, 0, 1],
    ]
    f = lambda t: solver.solve(waypoints, t)
    t0 = np.ones(len(waypoints) - 1)
    t = t0.copy()
    LR = 0.005
    for i in range(10):
        dt = numerical_grad(f, t)
        print("numerical:")
        print(dt)
        t = t - LR * dt
        t = np.clip(t, 0.1, None)
        t = t + (np.sum(t0) - np.sum(t)) / len(t)

    solver.solve(waypoints, t)
    solver.show_path()
