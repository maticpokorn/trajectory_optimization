import numpy as np
from constraint_builder import (
    LinearConstraintBuilder,
    GradLinearConstraintBuilder,
    SecondOrderDerivativeLinearConstraintBuilder,
)
from fast_derivative_constraint_builer import (
    EqualityConstraintBuilder1st,
    EqualityConstraintBuilder2nd,
)
from fast_constraint_builder import EqualityConstraintBuilder
from solver import TimestampSolver
from scipy.optimize import check_grad


class GradTester:
    def __init__(self, f, grad):
        self.f = f
        self.grad = grad

    def test(self, x):
        numerical_grad = self.numerical_grad(x)
        grad = self.grad(x)
        error = np.sum(np.abs(grad - numerical_grad))
        # print(grad)
        # print(numerical_grad)
        print(f"error: {error}")
        print(
            f"relative error: {(np.sum(abs(grad - numerical_grad)) / np.sum(abs(grad)))}"
        )
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


def test_2nd_derivative(x, t):
    dA, db = GradLinearConstraintBuilder([1, 2], 3, 2).build([1])


if __name__ == "__main__":
    waypoints = np.array([[1, 0, 0], [2, -1, 1], [3, 0, 2], [2, 1, 2]])
    t = np.array([1.0, 0.5, 1.5])

    solver = TimestampSolver(7, 4, 4, 3)

    """grad_tester = GradTester(
        lambda t: solver.solve(waypoints, t), lambda t: solver.grad(waypoints, t)
    )
    grad_tester.test(t)

    grad_tester = GradTester(
        lambda t: LinearConstraintBuilder(waypoints[:, 0], 7, 4).build(t)[0],
        lambda t: GradLinearConstraintBuilder(waypoints[:, 0], 7, 4).build(t),
    )
    grad_tester.test(t)

    grad_tester = GradTester(
        lambda t: solver.P(t),
        lambda t: solver.dP(t),
    )
    grad_tester.test(t)"""
    A = lambda t: LinearConstraintBuilder(waypoints[:, 0], 3, 2).build(t)
    dAs = lambda t: GradLinearConstraintBuilder(waypoints[:, 0], 3, 2).build(t)
    ddAs = lambda t: SecondOrderDerivativeLinearConstraintBuilder(
        waypoints[:, 0], 3, 2
    ).build(t)

    A_ = lambda t: EqualityConstraintBuilder(waypoints[:, 0], 3, 2).build(t).matrix()[0]
    dAs_ = lambda t: np.array(
        EqualityConstraintBuilder1st(waypoints[:, 0], 3, 2).build(t).matrices()
    )
    ddAs_ = (
        lambda t: EqualityConstraintBuilder2nd(waypoints[:, 0], 3, 2)
        .build(t)
        .matrices()
    )

    # t = [0.6, 0.5]
    print(A_(t))
    print(A(t))
    tester = GradTester(A_, dAs_)
    print("num")
    print(tester.numerical_grad(t))
    print("analytic")
    print(tester.grad(t))
    tester.test(t)
    # print(dAs(t)[1])
    # print(dAs_(t)[1])
    """
    dP = lambda t: TimestampSolver(3, 2, 2, 3).dPT(t)
    ddP = lambda t: TimestampSolver(3, 2, 2, 3).ddPT(t)

    print(dP(1.0))
    print(ddP(1.0))

    t = [0.01]
    print("tester")
    tester = GradTester(dP, ddP)
    print("num")
    print(tester.numerical_grad(t))
    print("analytic")
    print(tester.grad(t[0]))
    # tester.test(t)"""
    """print("first order")
    print(dAs(t))
    print("second order")
    print(ddAs(t))

    print("tester")
    tester = GradTester(dAs, ddAs)
    print("num")
    print(tester.numerical_grad(t))
    print("analytic")
    print(tester.grad(t))
    # tester.test(t)"""
