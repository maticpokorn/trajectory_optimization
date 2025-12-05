import numpy as np
import math


class EqualityDerivativeConstraintBuilder:
    def __init__(self, n, d, q):
        self.n = n
        self.d = d
        self.q = q
        self.fact = np.array(
            [math.factorial(k) for k in range(self.d + 1)], dtype=float
        )
        self.constraints = []

    def build(self, t):
        self.constraints.append(np.zeros((self.q + 1, self.d + 1)))
        for i in range(self.n - 1):
            self.constraints.append(self.continuity_constraint(i, t))
        self.constraints.append(self.end_constraint(t))
        return self

    def matrices(self):
        matrices = []
        n = self.q + 2
        m = self.d + 1
        i = n - 1
        j = 0
        for c in self.constraints[1:-1]:
            A = np.zeros(
                ((self.n + 1) * (self.q + 1) + (self.n - 1), self.n * (self.d + 1))
            )
            A[i : i + n, j : j + m] = c
            matrices.append(A)
            i += n
            j += m
        A = np.zeros(
            ((self.n + 1) * (self.q + 1) + (self.n - 1), self.n * (self.d + 1))
        )
        A[-(self.q + 1) :, -(self.d + 1) :] = self.constraints[-1]
        matrices.append(A)
        return matrices

    def continuity_constraint(self, i, t):
        return None

    def end_constraint(self, t):
        return None


class EqualityConstraintBuilder1st(EqualityDerivativeConstraintBuilder):

    def continuity_constraint(self, i, t):
        AT = np.zeros((self.q + 2, self.d + 1))
        for j in range(self.q + 1):
            k_vals = np.arange(j + 1, self.d + 1)
            AT[j, k_vals] = (
                self.fact[k_vals]
                / self.fact[k_vals - (j + 1)]
                * (t[i] ** (k_vals - (j + 1)))
            )
        return AT

    def end_constraint(self, t):
        A = np.zeros((self.q + 1, self.d + 1))
        for j in range(self.q + 1):
            k_vals = np.arange(j + 1, self.d + 1)
            A[j, k_vals] = (
                self.fact[k_vals]
                / self.fact[k_vals - (j + 1)]
                * (t[-1] ** (k_vals - (j + 1)))
            )
        return A


class EqualityConstraintBuilder2nd(EqualityDerivativeConstraintBuilder):

    def continuity_constraint(self, i, t):
        AT = np.zeros((self.q + 2, self.d + 1))
        for j in range(self.q + 1):
            k_vals = np.arange(j + 2, self.d + 1)
            AT[j, k_vals] = (
                self.fact[k_vals]
                / self.fact[k_vals - (j + 2)]
                * (t[i] ** (k_vals - (j + 2)))
            )
        return AT

    def end_constraint(self, t):
        A = np.zeros((self.q + 1, self.d + 1))
        for j in range(self.q + 1):
            k_vals = np.arange(j + 2, self.d + 1)
            A[j, k_vals] = (
                self.fact[k_vals]
                / self.fact[k_vals - (j + 2)]
                * (t[-1] ** (k_vals - (j + 2)))
            )
        return A
