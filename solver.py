import numpy as np


from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import block_diag
from constraint_builder import LinearConstraintBuilder, GradLinearConstraintBuilder
from time import time
from scipy.sparse.linalg import spsolve, minres
from scipy.sparse import csr_matrix


class Solver:
    def __init__(self, d, r, q, dimensions=2):
        # degree of polynomials
        self.d = d
        # optimize r-th derivative
        self.r = r
        # ensure continuity of up to q-th derivative
        self.q = q
        # dimensions (2 or 3)
        self.dimensions = dimensions
        self.waypoints = None
        self.timestamps = None
        self.obj = None

    def solve(self):
        return None

    def show_path(self):
        if self.dimensions == 2:
            self.show_path_2d()
        if self.dimensions == 3:
            self.show_path_3d()

    def show_path_2d(self):
        coeffs = self.result
        _, ax = plt.subplots()
        # ax.plot(self.waypoints.T[0], self.waypoints.T[1], color="lime")
        ax.scatter(self.waypoints.T[0], self.waypoints.T[1], color="red")
        for p, label in zip(self.waypoints, self.timestamps):
            plt.text(
                p[0], p[1], str(round(label, 2)), fontsize=9, ha="right", va="bottom"
            )
        x = []
        y = []
        v = []
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            dpx = np.poly1d((coeffs[0, i] * np.arange(self.d + 1))[1:][::-1])
            dpy = np.poly1d((coeffs[1, i] * np.arange(self.d + 1))[1:][::-1])
            t = np.linspace(0, self.timestamps[i + 1] - self.timestamps[i], 100)
            x.append(px(t))
            y.append(py(t))
            dx = dpx(t)
            dy = dpy(t)
            v.append(np.sqrt((dx**2 + dy**2))[:-1])

        x = np.hstack(x)
        y = np.hstack(y)
        v = np.hstack(v)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="plasma", array=v / np.max(v), linewidth=2)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="velocity")
        plt.axis("equal")
        plt.show()

    def show_path_3d(self):
        coeffs = self.result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        """ax.plot(
            self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2], color="lime"
        )"""
        ax.scatter(
            self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2], color="red"
        )
        for i, p in enumerate(self.waypoints):
            ax.text(
                p[0],
                p[1],
                p[2],
                str(round(np.sum(self.t[:i]), 2)),
                fontsize=9,
                ha="right",
                va="bottom",
            )
        x = []
        y = []
        z = []
        v = []
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            pz = np.poly1d(coeffs[2, i][::-1])
            dpx = np.poly1d((coeffs[0, i] * np.arange(self.d + 1))[1:][::-1])
            dpy = np.poly1d((coeffs[1, i] * np.arange(self.d + 1))[1:][::-1])
            dpz = np.poly1d((coeffs[2, i] * np.arange(self.d + 1))[1:][::-1])
            T = np.linspace(0, self.t[i], 100)
            x.append(px(T))
            y.append(py(T))
            z.append(pz(T))
            dx = dpx(T)
            dy = dpy(T)
            dz = dpz(T)
            v.append(np.sqrt((dx**2 + dy**2 + dz**2))[:-1])

        x = np.hstack(x)
        y = np.hstack(y)
        z = np.hstack(z)
        v = np.hstack(v)
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap="plasma", array=v / np.max(v), linewidth=2)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="velocity")
        plt.axis("equal")
        plt.show()


class TimestampSolver(Solver):
    def __init__(self, d, r, q, dimensions):
        super().__init__(d, r, q, dimensions)

    def solve(self, waypts, t):
        self.waypoints = np.array(waypts)
        self.t = np.array(t)
        self.n_segments = len(t)

        self.obj = 0
        self.nu = []
        self.x = []

        result = np.zeros((self.dimensions, self.n_segments, self.d + 1))
        P = self.P(self.t)
        q = np.zeros(self.n_segments * (self.d + 1))
        for dim in range(self.dimensions):
            A, b, _ = LinearConstraintBuilder(
                self.waypoints[:, dim], self.d, self.q
            ).build(self.t)
            # self.test_matrices(P, A, b)
            sol = qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
            res_dim = np.array(sol["x"]).reshape((self.n_segments, self.d + 1))
            self.x.append(np.array(sol["x"])[:, 0])
            self.nu.append(np.array(sol["y"])[:, 0])
            result[dim, :, :] = res_dim
            self.result = result
            self.obj += sol["primal objective"]
        return self.obj

    def solve_with(self, P, q, A, b, solver_type="cvxopt"):
        if solver_type == "cvxopt":
            sol = qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
            x = np.array(sol["x"])[:, 0]
            nu = np.array(sol["y"])[:, 0]
            res_dim = np.array(sol["x"]).reshape((self.n_segments, self.d + 1))
            obj = sol["primal objective"]

    def test_matrices(self, P, A, b):
        n = P.shape[0]
        print("A shape: ", A.shape)
        print("rank(A) =", np.linalg.matrix_rank(A))
        print("rank([A|b]) =", np.linalg.matrix_rank(np.hstack([A, b.reshape(-1, 1)])))
        print(
            "Rank([P;A;G]):",
            (np.linalg.matrix_rank(np.vstack([P, A]))),
        )
        print("n =", n)

    def grad(self, waypts, t):
        n_segments = len(t)
        grad = np.zeros((self.dimensions, n_segments))
        P = self.P(t)
        dPs = self.dP(t)
        matrix_multiplication_time = 0
        system_solve_time = 0
        constraint_building_time = 0
        for dim in range(self.dimensions):
            start = time()
            A, b, independent = LinearConstraintBuilder(
                waypts[:, dim], self.d, self.q
            ).build(t)
            # print(independent)
            dAs = GradLinearConstraintBuilder(waypts[:, dim], self.d, self.q).build(t)
            constraint_building_time += time() - start

            x = self.x[dim]
            y = self.nu[dim]

            m, n = A.shape
            K = csr_matrix(np.block([[P, A.T], [A, np.zeros((m, m))]]))
            """print(
                f"share of nonzeros: {round(np.mean(K != 0), 4)}, number of nonzeros: {np.sum(K != 0)}, nonzeros / N: {round(np.sum(K != 0)/n_segments, 4)}"
            )"""
            for i in range(n_segments):
                dP = dPs[i]
                dA = dAs[i][independent]
                start = time()
                rhs = -np.concatenate((dP @ x + A.T @ y, dA @ x))
                matrix_multiplication_time += time() - start
                start = time()
                dxy = spsolve(A=K, b=rhs)
                system_solve_time += time() - start
                dx = dxy[: (self.d + 1) * n_segments][:, np.newaxis]

                dF = dx.T @ P @ x + 1 / 2 * x.T @ dP @ x
                grad[dim, i] = dF

        # print(f"time for matrix multiplication: {matrix_multiplication_time}")
        # print(f"time for solving system: {system_solve_time}")
        # print(f"time for building constraints: {constraint_building_time}")
        return np.sum(grad, axis=0)

    def P(self, t):
        PTs = [self.PT(T) for T in t]
        P = block_diag(*PTs)  # + 1e-5 * np.eye(self.n_segments * (self.d + 1))
        return P

    def PT(self, T):
        # quadratic cost matrix, corresponding to the segment ending with timestamp T
        PT = np.zeros((self.d + 1, self.d + 1))
        for l in range(self.r, self.d + 1):
            for k in range(l, self.d + 1):
                q = (
                    np.prod([(l - m) * (k - m) for m in range(self.r)])
                    * T ** (l + k - 2 * self.r + 1)
                    / (l + k - 2 * self.r + 1)
                )
                PT[l, k] = q
                PT[k, l] = q
        return PT

    def dP(self, t):
        dPs = []
        n = len(t)
        for i in range(n):
            dP = np.zeros((n * (self.d + 1), n * (self.d + 1)))
            dPT = self.dPT(t[i])
            dP[
                i * (self.d + 1) : (i + 1) * (self.d + 1),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = dPT
            dPs.append(dP)
        return np.array(dPs)

    def dPT(self, T):
        dPT = np.zeros((self.d + 1, self.d + 1))
        for l in range(self.r, self.d + 1):
            for k in range(l, self.d + 1):
                q = np.prod([(l - m) * (k - m) for m in range(self.r)]) * T ** (
                    l + k - 2 * self.r
                )
                dPT[l, k] = q
                dPT[k, l] = q
        return dPT


class NoTimestampSolver(Solver):
    def __init__(self, d, r, q, dimensions, tol=0.001):
        super().__init__(d, r, q, dimensions)
        self.timestamp_solver = TimestampSolver(d, r, q, dimensions)
        self.tol = tol

    def solve(self, waypoints, T, lr=0.1):
        self.waypoints = np.array(waypoints)
        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)
        n = self.waypoints.shape[0] - 1
        t = np.random.rand(n)
        t = T / np.sum(t) * t
        prev_obj = 1e12
        dif = 1e12
        i = 0
        while dif > self.tol and i < 1000:
            obj = self.timestamp_solver.solve(self.waypoints, t)
            grad = self.timestamp_solver.grad(self.waypoints, t)
            grad = grad / np.linalg.norm(grad)
            grad = grad - np.mean(grad)
            t -= lr * grad
            # t *= T / np.sum(t)
            # print(t)
            dif = abs(prev_obj - obj)
            prev_obj = obj
            i += 1
            print(obj)
        print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj
