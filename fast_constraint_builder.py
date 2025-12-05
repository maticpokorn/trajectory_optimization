import numpy as np
import math


class EqualityConstraintBuilder:

    def __init__(self, x, d, q):
        self.n = len(x) - 1
        self.x = x
        self.d = d
        self.q = q
        self.fact = np.array(
            [math.factorial(k) for k in range(self.d + 1)], dtype=float
        )
        self.blocks = []
        self.bs = []

    def build(self, t):
        A_start, b = self.start_constraint(t)
        self.blocks.append((None, A_start))
        self.bs.append(b)
        for i in range(1, self.n + 1):
            AT, A0, b = self.continuity_constraint(i, t)
            self.blocks.append((AT, A0))
            self.bs.append(b)
        A_end, b = self.end_constraint(t)
        self.blocks.append((A_end, None))
        self.bs.append(b)
        return self

    def start_constraint(self, t):
        A = np.zeros((self.q + 1, self.d + 1))
        b = np.zeros(self.q + 1)
        for j in range(self.q + 1):
            A[j, j] = self.fact[j]
        b[0] = self.x[0]
        return A, b

    def continuity_constraint(self, i, t):
        AT = np.zeros((self.q + 2, self.d + 1))
        A0 = np.zeros((self.q + 2, self.d + 1))
        b = np.zeros(self.q + 2)
        for j in range(self.q + 1):
            A0[j, j] = -self.fact[j]
            k_vals = np.arange(j, self.d + 1)
            AT[j, k_vals] = (
                self.fact[k_vals] / self.fact[k_vals - j] * (t[i - 1] ** (k_vals - j))
            )
        A0[self.q + 1, 0] = 1
        b[self.q + 1] = self.x[i]
        return AT, A0, b

    def end_constraint(self, t):
        A = np.zeros((self.q + 1, self.d + 1))
        b = np.zeros(self.q + 1)
        for j in range(self.q + 1):
            k_vals = np.arange(j, self.d + 1)
            A[j, k_vals] = (
                self.fact[k_vals] / self.fact[k_vals - j] * (t[-1] ** (k_vals - j))
            )
        b[0] = self.x[-1]
        return A, b

    def matrix(self):
        A = np.zeros(
            ((self.n + 1) * (self.q + 1) + (self.n - 1), self.n * (self.d + 1))
        )
        b = np.zeros((self.n + 1) * (self.q + 1) + (self.n - 1))
        n = self.q + 2
        m = self.d + 1
        A[: n - 1, :m] = self.blocks[0][1]
        b[: n - 1] = self.bs[0]
        i = n - 1
        j = 0
        for k in range(1, self.n):
            A[i : i + n, j : j + m] = self.blocks[k][0]
            A[i : i + n, j + m : j + 2 * m] = self.blocks[k][1]
            b[i : i + n] = self.bs[k]
            i += n
            j += m
        A[-(self.q + 1) :, -(self.d + 1) :] = self.blocks[-1][0]
        b[-(self.q + 1) :] = self.bs[-1]
        return A, b
