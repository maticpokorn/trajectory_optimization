from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, lsqr
import numpy as np


class NullSpaceSolver:

    def __init__(self, P, A):
        self.P = csr_matrix(P)
        self.A = csr_matrix(A)
        self.AT = csr_matrix(A.T)
        self.Z = self.calc_null_space(A)
        self.P_ = csr_matrix(self.Z.T @ self.P @ self.Z)
        print("max: ", np.max(self.A @ self.Z))

    def calc_null_space(self, A):
        return null_space(A)

    def solve(
        self,
        f,
        g,
    ):
        # Solve A x_hat = g to find a particular solution for constraint
        x_hat = lsqr(self.A, g, atol=1e-12, btol=1e-12, iter_lim=10000)[0]
        # Right-hand side for reduced system
        rhs = self.Z.T @ (f - self.P @ x_hat)
        # Solve Z^T P Z z = rhs
        z = spsolve(self.P_, rhs)
        # Full x
        x = x_hat + self.Z @ z
        # Solve for y from A.T y = f - P x
        y = lsqr(self.AT, f - self.P @ x, atol=1e-12, btol=1e-12, iter_lim=10000)[0]
        print(f"error: {np.max(self.A.T @ y - (f - self.P @ x))}")
        return x, y
