import numpy as np
from constraint_builder import LinearConstraintBuilder, GradLinearConstraintBuilder
from solver import TimestampSolver


class GradTester:
    def __init__(self, f, grad):
        self.f = f
        self.grad = grad

    def test(self, x):
        numerical_grad = self.numerical_grad(x)
        grad = self.grad(x)
        error = np.sum(np.abs(grad - numerical_grad))
        print(f"error: {error}")
        return error

    def numerical_grad(self, x, eps=1e-6):
        """
        Compute gradient of scalar function f at x using central differences
        """
        x = np.array(x, dtype=float)
        grad = []

        for i in range(len(x)):
            x_eps_plus = x.copy()
            x_eps_minus = x.copy()
            x_eps_plus[i] += eps
            x_eps_minus[i] -= eps
            grad.append((self.f(x_eps_plus) - self.f(x_eps_minus)) / (2 * eps))

        return np.array(grad)


if __name__ == "__main__":
    waypoints = np.array(
        [[1, 0, 0], [2, -1, 1], [2, 0, 1], [1, 1, 1], [0, 0, 1], [0, 2, 2]]
    )
    t = np.array([0.5, 0.66, 1.1, 0.9, 0.75])

    solver = TimestampSolver(7, 4, 4, 3)

    tester = GradTester(
        lambda t: solver.solve(waypoints, t), lambda t: solver.grad(waypoints, t)
    )
    print(tester.numerical_grad(t))
    print(tester.grad(t))
    tester.test(t)
