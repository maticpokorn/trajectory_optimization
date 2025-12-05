import numpy as np
import math
import sympy
from scipy.sparse import csr_matrix
from scipy.linalg import qr

"""
structure of polynomial coefficients vector
p_ij ... coefficient at t**j of i-th segment
[p_00, p_01, ..., p_01, p_11, ..., p_00, ..., p_00]

"""


class LinearConstraintBuilder:
    """
    builds constraint matrix A and constraint vector b
    """

    def __init__(self, x, d, q):
        self.n = len(x) - 1
        self.x = x
        self.d = d
        self.q = q
        self.fact = np.array(
            [math.factorial(k) for k in range(self.d + 1)], dtype=float
        )

    def build(self, t):
        """
        Builds constraints
        """
        constraints = []
        # add all waypoint constraints
        for i in range(1, self.n):
            # inner waypoints
            constraints.append(self.waypoint_constraint(i))
            # inner waypoints derivatives
            for j in range(self.q + 1):
                constraints.append(self.continuity_constraint(i, j, t))

        # start waypoint
        constraints.append(self.start_constraint(0, self.x[0]))
        # end waypoint
        constraints.append(self.end_constraint(0, self.x[-1], t))
        # start and end waypoints derivatives
        for j in range(1, self.q + 1):
            constraints.append(self.start_constraint(j, 0))
            constraints.append(self.end_constraint(j, 0, t))

        A, b = self.unpack(constraints)
        # independent = self.independent(A, b)
        independent = np.arange(A.shape[0])
        return A[independent], b[independent], independent

    def unpack(self, constraints):
        A = np.zeros((len(constraints), self.n * (self.d + 1)))
        b = np.zeros(len(constraints))
        for i, constraint in enumerate(constraints):
            A[i] = constraint.a
            b[i] = constraint.b
        return A, b

    def independent(self, A, b):
        _, inds = sympy.Matrix(np.hstack((A, b.reshape(-1, 1)))).T.rref()
        return list(inds)

    def waypoint_constraint(self, i, start=True):
        """
            Defines waypoint constraints (trajectory has to go through waypoint).

        Args:
            i (int): enforcing trajectory through i-th waypoint.
            start (bool): enforcing at start or end of the interval.

        Returns:
            numpy array: row with coefficients.
            float: constraint value (i-th waypoint).
        """
        if start:
            a = np.zeros(self.n * (self.d + 1))
            a[i * (self.d + 1)] = 1
            return Constraint(a, self.x[i])

    def continuity_constraint(self, i, j, t):
        """
            Defines derivative continuity constraints.

        Args:
            i (int): enforcing continuity at i-th waypoint.
            j (int): enforcing continuity of j-th derivative.

        Returns:
            numpy array: row with coefficients.
            float: constraint value (0).
        """
        # precompute factorials up to d
        # fact = np.array([math.factorial(k) for k in range(self.d + 1)], dtype=float)

        # a0
        a0 = np.zeros(self.d + 1)
        a0[j] = self.fact[j]

        # at
        k_vals = np.arange(j, self.d + 1)
        at = np.zeros(self.d + 1)
        at[k_vals] = (
            self.fact[k_vals] / self.fact[k_vals - j] * (t[i - 1] ** (k_vals - j))
        )

        # a
        a = np.zeros(self.n * (self.d + 1))
        a[(i - 1) * (self.d + 1) : i * (self.d + 1)] = at
        a[i * (self.d + 1) : (i + 1) * (self.d + 1)] = -a0
        return Constraint(a, 0)

    def start_constraint(self, j, value):
        """
            Defines inital values for j-th derivative.

        Args:
            j (int): enforcing start value of j-th derivative.
            value (float): start value of j-th derivative.

        Returns:
            numpy array: row with coefficients.
            float: constraint value (j-th derivative).
        """
        a = np.zeros(self.n * (self.d + 1))
        a[j] = math.factorial(j)
        return Constraint(a, value)

    def end_constraint(self, j, value, t):
        """
            Defines stopping values for j-th derivative.

        Args:
            j (int): enforcing stop value of j-th derivative.

        Returns:
            numpy array: row with coefficients.
            float: constraint value (j-th derivative).
        """
        # precompute factorials up to d
        # fact = np.array([math.factorial(k) for k in range(self.d + 1)], dtype=float)

        # at
        k_vals = np.arange(j, self.d + 1)
        at = np.zeros(self.d + 1)
        at[k_vals] = self.fact[k_vals] / self.fact[k_vals - j] * (t[-1] ** (k_vals - j))

        # a
        a = np.zeros(self.n * (self.d + 1))
        a[-(self.d + 1) :] = at
        return Constraint(a, value)

    def corridor_constraint(i, T):
        return None

    def ball_constraint(i, R):
        return None


class Constraint:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class GradLinearConstraintBuilder:
    """
    builds gradient of constraint matrix A w.r.t. t
    """

    def __init__(self, x, d, q):
        self.n = len(x) - 1
        self.x = x
        self.d = d
        self.q = q

    def build_k(self, t, k):
        constraints = []
        # add all waypoint constraints
        for i in range(1, self.n):
            # inner waypoints

            # constraints.append(self.waypoint_constraint(i))
            constraints.append(self.empty_constraint)
            # inner waypoints derivatives
            for j in range(self.q + 1):
                if k == i - 1:
                    constraints.append(self.continuity_constraint(i, j, t))
                else:
                    constraints.append(self.empty_constraint)

        # start waypoint
        # constraints.append(self.start_constraint(0, self.x[0]))
        constraints.append(self.empty_constraint)
        # end waypoint
        if k == self.n - 1:
            constraints.append(self.end_constraint(0, self.x[-1], t))
        else:
            constraints.append(self.empty_constraint)

        # start and end waypoints derivatives
        for j in range(1, self.q + 1):
            # constraints.append(self.start_constraint(j, 0))
            constraints.append(self.empty_constraint)
            if k == self.n - 1:
                constraints.append(self.end_constraint(j, 0, t))
            else:
                constraints.append(self.empty_constraint)

        return self.unpack(constraints)[0]

    def build(self, t):
        """
        Builds constraints gradient of build(t) for LinearConstraintBuilder
        """
        grad = []
        for k in range(self.n):
            constraints = []
            # add all waypoint constraints
            for i in range(1, self.n):
                # inner waypoints

                # constraints.append(self.waypoint_constraint(i))
                constraints.append(self.empty_constraint)
                # inner waypoints derivatives
                for j in range(self.q + 1):
                    if k == i - 1:
                        constraints.append(self.continuity_constraint(i, j, t))
                    else:
                        constraints.append(self.empty_constraint)

            # start waypoint
            # constraints.append(self.start_constraint(0, self.x[0]))
            constraints.append(self.empty_constraint)
            # end waypoint
            if k == self.n - 1:
                constraints.append(self.end_constraint(0, self.x[-1], t))
            else:
                constraints.append(self.empty_constraint)

            # start and end waypoints derivatives
            for j in range(1, self.q + 1):
                # constraints.append(self.start_constraint(j, 0))
                constraints.append(self.empty_constraint)
                if k == self.n - 1:
                    constraints.append(self.end_constraint(j, 0, t))
                else:
                    constraints.append(self.empty_constraint)
            grad.append(self.unpack(constraints)[0])

        return np.array(grad)

    def unpack(self, constraints):
        A = np.zeros((len(constraints), self.n * (self.d + 1)))
        b = np.zeros(len(constraints))
        for i, constraint in enumerate(constraints):
            A[i] = constraint.a
            b[i] = constraint.b
        return A, b

    def __init__(self, x, d, q):
        self.n = len(x) - 1
        self.x = x
        self.d = d
        self.q = q
        self.empty_constraint = Constraint(np.zeros(self.n * (self.d + 1)), 0)
        self.fact = np.array(
            [math.factorial(k) for k in range(self.d + 1)], dtype=float
        )

    def waypoint_constraint(self, i, start=True):
        if start:
            a = np.zeros(self.n * (self.d + 1))
            return Constraint(a, self.x[i])

    def continuity_constraint(self, i, j, t):
        """
            Defines derivative continuity constraints.

        Args:
            i (int): enforcing continuity at i-th waypoint.
            j (int): enforcing continuity of j-th derivative.
            k (int): partial derivative w.r.t. t_k

        Returns:
            numpy array: row with coefficients.
            float: constraint value (0).
        """
        # precompute factorials up to d
        # fact = np.array([math.factorial(k) for k in range(self.d + 1)], dtype=float)

        # a0
        # a0 = np.zeros(self.d + 1)
        # a0[j] = fact[j]

        # at
        k_vals = np.arange(j + 1, self.d + 1)
        at = np.zeros(self.d + 1)
        at[k_vals] = (
            self.fact[k_vals]
            / self.fact[k_vals - (j + 1)]
            * (t[i - 1] ** (k_vals - (j + 1)))
        )

        # a
        a = np.zeros(self.n * (self.d + 1))
        a[(i - 1) * (self.d + 1) : i * (self.d + 1)] = at
        # a[i * (self.d + 1) : (i + 1) * (self.d + 1)] = -a0
        return Constraint(a, 0)

    def end_constraint(self, j, value, t):
        """
            Defines stopping values for j-th derivative.

        Args:
            j (int): enforcing stop value of j-th derivative.

        Returns:
            numpy array: row with coefficients.
            float: constraint value (j-th derivative).
        """
        # precompute factorials up to d
        # fact = np.array([math.factorial(k) for k in range(self.d + 1)], dtype=float)

        # at
        k_vals = np.arange(j + 1, self.d + 1)
        at = np.zeros(self.d + 1)
        at[k_vals] = (
            self.fact[k_vals]
            / self.fact[k_vals - (j + 1)]
            * (t[-1] ** (k_vals - (j + 1)))
        )

        # a
        a = np.zeros(self.n * (self.d + 1))
        a[-(self.d + 1) :] = at
        return Constraint(a, 0)


class SecondOrderDerivativeLinearConstraintBuilder:

    def __init__(self, x, d, q):
        self.n = len(x) - 1
        self.x = x
        self.d = d
        self.q = q
        self.empty_constraint = Constraint(np.zeros(self.n * (self.d + 1)), 0)
        self.fact = np.array(
            [math.factorial(k) for k in range(self.d + 1)], dtype=float
        )

    def build(self, t):
        """
        Builds constraints diagonal of Hessian of of build(t) for LinearConstraintBuilder
        """
        grad = []
        for k in range(self.n):
            constraints = []
            # add all waypoint constraints
            for i in range(1, self.n):
                # inner waypoints

                # constraints.append(self.waypoint_constraint(i))
                constraints.append(self.empty_constraint)
                # inner waypoints derivatives
                for j in range(self.q + 1):
                    if k == i - 1:
                        constraints.append(self.continuity_constraint(i, j, t))
                    else:
                        constraints.append(self.empty_constraint)

            # start waypoint
            # constraints.append(self.start_constraint(0, self.x[0]))
            constraints.append(self.empty_constraint)
            # end waypoint
            if k == self.n - 1:
                constraints.append(self.end_constraint(0, self.x[-1], t))
            else:
                constraints.append(self.empty_constraint)

            # start and end waypoints derivatives
            for j in range(1, self.q + 1):
                # constraints.append(self.start_constraint(j, 0))
                constraints.append(self.empty_constraint)
                if k == self.n - 1:
                    constraints.append(self.end_constraint(j, 0, t))
                else:
                    constraints.append(self.empty_constraint)
            grad.append(self.unpack(constraints)[0])

        return np.array(grad)

    def unpack(self, constraints):
        A = np.zeros((len(constraints), self.n * (self.d + 1)))
        b = np.zeros(len(constraints))
        for i, constraint in enumerate(constraints):
            A[i] = constraint.a
            b[i] = constraint.b
        return A, b

    def waypoint_constraint(self, i, start=True):
        if start:
            a = np.zeros(self.n * (self.d + 1))
            return Constraint(a, self.x[i])

    def continuity_constraint(self, i, j, t):
        """
            Defines derivative continuity constraints.

        Args:
            i (int): enforcing continuity at i-th waypoint.
            j (int): enforcing continuity of j-th derivative.
            k (int): partial derivative w.r.t. t_k

        Returns:
            numpy array: row with coefficients.
            float: constraint value (0).
        """
        # precompute factorials up to d
        # fact = np.array([math.factorial(k) for k in range(self.d + 1)], dtype=float)

        # a0
        # a0 = np.zeros(self.d + 1)
        # a0[j] = fact[j]

        # at
        k_vals = np.arange(j + 2, self.d + 1)
        at = np.zeros(self.d + 1)
        at[k_vals] = (
            self.fact[k_vals]
            / self.fact[k_vals - (j + 2)]
            * (t[i - 1] ** (k_vals - (j + 2)))
        )

        # a
        a = np.zeros(self.n * (self.d + 1))
        a[(i - 1) * (self.d + 1) : i * (self.d + 1)] = at
        # a[i * (self.d + 1) : (i + 1) * (self.d + 1)] = -a0
        return Constraint(a, 0)

    def end_constraint(self, j, value, t):
        """
            Defines stopping values for j-th derivative.

        Args:
            j (int): enforcing stop value of j-th derivative.

        Returns:
            numpy array: row with coefficients.
            float: constraint value (j-th derivative).
        """
        # precompute factorials up to d
        # fact = np.array([math.factorial(k) for k in range(self.d + 1)], dtype=float)

        # at
        k_vals = np.arange(j + 2, self.d + 1)
        at = np.zeros(self.d + 1)
        at[k_vals] = (
            self.fact[k_vals]
            / self.fact[k_vals - (j + 2)]
            * (t[-1] ** (k_vals - (j + 2)))
        )

        # a
        a = np.zeros(self.n * (self.d + 1))
        a[-(self.d + 1) :] = at
        return Constraint(a, 0)
