from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
import matplotlib.pyplot as plt
import sympy
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.optimize import minimize, LinearConstraint, Bounds


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
        self.sol = None
        self.nu = None
        self.result = None
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
        ax.plot(self.waypoints.T[0], self.waypoints.T[1], color="lime")
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
        ax.plot(
            self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2], color="lime"
        )
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


class NoTimestampSolver(Solver):
    def __init__(self, d, r, q, dimensions):
        super().__init__(d, r, q, dimensions)
        self.timestamp_solver = TimestampSolver(d, r, q, dimensions)

    def solve(self, waypoints, T):
        self.waypoints = np.array(waypoints)
        n = self.waypoints.shape[0]
        timestamps0 = np.arange(1, n - 1) / (n - 1) * T
        print(timestamps0)
        lc = self.linear_constraint(T)
        f = lambda tstamps: self.timestamp_solver.solve(
            self.waypoints, np.hstack((np.array([0]), tstamps, np.array([T])))
        )
        res = minimize(
            f,
            timestamps0,
            method="trust-constr",
            constraints=[lc],
            tol=0.5,
        )
        self.timestamps = np.hstack((np.array([0]), res.x, np.array([T])))
        self.result = self.timestamp_solver.result
        self.obj = self.timestamp_solver.obj
        print(self.obj)
        return self.obj

    def solve_grad(self, waypoints, T):
        self.waypoints = np.array(waypoints)
        n = self.waypoints.shape[0]
        tstamps = np.arange(0, n) / n * T

        segment_times = tstamps[1:] - tstamps[:-1]
        f = lambda segment_times: self.timestamp_solver.solve(
            self.waypoints, np.hstack((np.array([0]), np.cumsum(segment_times)))
        )
        df = lambda segment_times: (
            self.timestamp_solver.cost_derivative(segment_times)
        )
        reg = lambda segment_times: 2 * (np.cumsum(segment_times) - T)
        lambda_ = 1e-4
        for i in range(200):
            print(f"{i}: {f(segment_times)}, {np.sum(segment_times)}")
            grad = df(segment_times)
            segment_times = segment_times + lambda_ * grad

        self.timestamps = self.timestamp_solver.timestamps
        self.result = self.timestamp_solver.result
        self.obj = self.timestamp_solver.obj

        return self.obj

    def solve_grad2(self, waypoints, T):
        self.waypoints = np.array(waypoints)
        n = self.waypoints.shape[0]
        segment_times = np.ones(n - 1) * (T / n)
        f = lambda segment_times: self.timestamp_solver.solve(
            self.waypoints, np.hstack((np.array([0]), np.cumsum(segment_times)))
        )
        fr = (
            lambda segment_times: f(segment_times)
            + 20 * (np.sum(segment_times) - T) ** 2
        )
        df = lambda segment_times: -(
            self.timestamp_solver.cost_derivative(segment_times)
        )
        dfr = lambda segment_times: df(segment_times) - 2 * 10 * (
            np.sum(segment_times) - T
        )
        bounds = Bounds(np.zeros(n - 1), np.full(n - 1, np.inf))
        lc = self.linear_constraint(T)
        res = minimize(fr, segment_times, method="L-BFGS-B", bounds=bounds)

        self.timestamps = self.timestamp_solver.timestamps
        self.result = self.timestamp_solver.result
        self.obj = self.timestamp_solver.obj

        return self.obj

    def linear_constraint(self, T):
        n = self.waypoints.shape[0] - 2
        A = np.eye(n)
        A = np.vstack((A, np.zeros(n)))
        A[-1, -1] = 1
        for i in range(1, n):
            A[i, i - 1] = -1
        lb = np.zeros(n + 1)
        ub = np.full(n + 1, np.inf)
        ub[-1] = T
        linear_constraint = LinearConstraint(A, lb, ub)
        return linear_constraint

    def ineq_cons(self, T):
        cons_list = []
        cons_list.append({"type": "ineq", "fun": lambda x: x[0]})
        for i in range(self.waypoints.shape[0] - 3):
            cons_list.append({"type": "ineq", "fun": lambda x: x[i + 1] - x[i] - 0.1})
        cons_list.append({"type": "ineq", "fun": lambda x: T - x[-1]})
        return cons_list


1


class TimestampSolver(Solver):
    def __init__(self, d, r, q, dimensions):
        super().__init__(d, r, q, dimensions)

    def solve(self, waypoints, t):
        self.obj = 0
        # print(timestamps)
        waypoints = np.array(waypoints, dtype=np.float64)
        self.waypoints = waypoints
        t = np.array(t)
        # print(t)
        self.t = t
        n_segments = len(waypoints) - 1

        result = np.zeros((self.dimensions, n_segments, self.d + 1))
        dt = np.zeros(n_segments)
        for d in range(self.dimensions):
            P, q, A, b, inds = self.matrices(n_segments, d)
            dPs, dAs = self.derivative_matrices(n_segments, d, inds)
            K = self.K(P, A)
            sol = qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
            res_dim = np.array(sol["x"]).reshape((n_segments, self.d + 1))
            self.nu = sol["y"]
            result[d, :, :] = res_dim
            self.result = result
            self.obj += sol["primal objective"]
            p = sol["x"]
            print(A.shape)
            for i in range(n_segments):
                dQ = self.dQT(self.t[i])
                dA = self.dAT(self.t[i])
                print(dA)
                A_ = A[:, i * (self.d + 1) : (i + 1) * (self.d + 1)]
                _, inds = sympy.Matrix(A_).T.rref()
                A_ = A_[list(inds), :]
                nu = np.array(sol["y"])[list(inds)]
                print(nu)
                Q_ = self.QT(self.t[i]) + np.eye(self.d + 1) * 1e-6
                print(Q_.shape, A_.shape)
                K = self.K(Q_, A_)
                dp = self.dp(
                    K, dA, dQ, p[i * (self.d + 1) : (i + 1) * (self.d + 1)], nu
                )
                dT = dp.T @ Q_ @ p + p.T @ dQ @ p + p.T @ Q_ @ dp
                dt[i] += dT
                """dp_ = self.dp(K, dAs[i], dPs[i], p, sol["y"])
                dp = np.zeros((self.d + 1) * n_segments)[:, np.newaxis]
                dp[i * (self.d + 1) : (i + 1) * (self.d + 1)] = dp_[
                    i * (self.d + 1) : (i + 1) * (self.d + 1)
                ]
                dT = dp.T @ P @ p + p.T @ dPs[i] @ p + p.T @ P @ dp
                dt[i] += dT"""
        print(dt)

        print(self.obj)
        return self.obj

    def matrices(self, n, d):
        P = np.zeros((n * (self.d + 1), n * (self.d + 1)))
        q = np.zeros(n * (self.d + 1), dtype=np.float64)
        A = np.zeros(
            (
                (n - 1) * (self.q + 2) + 2 * (self.q + 1),
                n * (self.d + 1),
            ),
            dtype=np.float64,
        )
        b = np.zeros((n - 1) * (self.q + 2) + 2 * (self.q + 1), dtype=np.float64)
        A[: self.q + 1, : self.d + 1] = self.AT(0)
        b[0] = self.waypoints[0, d]
        for i in range(0, n - 1):
            QT = self.QT(self.t[i])
            P[
                i * (self.d + 1) : (i + 1) * (self.d + 1),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = QT
            A0 = self.AT(0)
            AT = self.AT(self.t[i])
            # define constraints for waypoint locations
            A[
                2 + i * (self.q + 2),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = AT[0]

            b[2 + i * (self.q + 2)] = self.waypoints[i + 1, d]
            # define constraints for equality of derivatives
            A[
                2 + i * (self.q + 2) + 1 : 2 + (i + 1) * (self.q + 2),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = AT
            A[
                2 + i * (self.q + 2) + 1 : 2 + (i + 1) * (self.q + 2),
                (i + 1) * (self.d + 1) : (i + 2) * (self.d + 1),
            ] = -A0
        QT = self.QT(self.t[-1])
        P[-(self.d + 1) :, -(self.d + 1) :] = QT
        A[-(self.q + 1) :, -(self.d + 1) :] = self.AT(self.t[-1])
        b[-(self.q + 1)] = self.waypoints[-1, d]

        # remove linearly dependent rows
        _, inds = sympy.Matrix(A).T.rref()
        A = A[list(inds)]
        b = b[list(inds)]
        # for numerical stability add identity matrix times a small scalar for regularization
        # print(A)
        P = P + 1e-5 * np.eye(n * (self.d + 1))
        b = b.reshape(-1, 1)
        return P, q, A, b, inds

    def dp(self, K, dA, dP, p, nu):
        print(K.shape, dA.shape, dP.shape, nu.shape)
        return -(np.linalg.inv(K) @ np.block([[dP @ p + dA.T @ nu], [dA @ p]]))[
            : (self.d + 1) * (len(self.waypoints) - 1)
        ]

    def K(self, P, A):
        return np.block([[P, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])

    def derivative_matrices(self, n, d, inds):
        dPs = []
        dAs = []
        for i in range(0, n - 1):
            dP = np.zeros((n * (self.d + 1), n * (self.d + 1)))
            dA = np.zeros(
                (
                    (n - 1) * (self.q + 2) + 2 * (self.q + 1),
                    n * (self.d + 1),
                ),
                dtype=np.float64,
            )
            dQT = self.dQT(self.t[i])
            dP[
                i * (self.d + 1) : (i + 1) * (self.d + 1),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = dQT
            dAT = self.dAT(self.t[i])
            dA[
                2 + i * (self.q + 2) + 1 : 2 + (i + 1) * (self.q + 2),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = dAT

            dPs.append(dP)
            dA = dA[inds, :]
            dAs.append(dA)

        dP = np.zeros((n * (self.d + 1), n * (self.d + 1)))
        dA = np.zeros(
            (
                (n - 1) * (self.q + 2) + 2 * (self.q + 1),
                n * (self.d + 1),
            ),
            dtype=np.float64,
        )
        dQT = self.dQT(self.t[-1])
        dP[-(self.d + 1) :, -(self.d + 1) :] = dQT
        dA[-(self.q + 1) :, -(self.d + 1) :] = self.dAT(self.t[-1])
        dPs.append(dP)
        dA = dA[inds, :]
        dAs.append(dA)
        return dPs, dAs

    def QT(self, T):
        # quadratic cost matrix, corresponding to the segment ending with timestamp T
        QT = np.zeros((self.d + 1, self.d + 1))
        for l in range(self.r, self.d + 1):
            for k in range(l, self.d + 1):
                q = (
                    np.prod([(l - m) * (k - m) for m in range(self.r)])
                    * T ** (l + k - 2 * self.r + 1)
                    / (l + k - 2 * self.r + 1)
                )
                QT[l, k] = q
                QT[k, l] = q
        return QT

    def cost_derivative(self, segment_times):
        dP = np.zeros(segment_times.shape[0])
        for i in range(segment_times.shape[0]):
            dp = 0
            dQ = self.dQ(segment_times[i])
            for j in range(self.dimensions):
                p = self.result[j, i, :]
                dp += p.T @ dQ @ p
            dP[i] = dp
        return dP

    def dQT(self, T):
        # derivative of quad matrix w.r.t. T
        dQT = np.zeros((self.d + 1, self.d + 1))
        for l in range(self.r, self.d + 1):
            for k in range(l, self.d + 1):
                q = np.prod([(l - m) * (k - m) for m in range(self.r)]) * T ** (
                    l + k - 2 * self.r
                )
                dQT[l, k] = q
                dQT[k, l] = q
        return dQT

    def AT(self, t):
        At = np.zeros((self.q + 1, self.d + 1))
        for i in range(self.q + 1):
            for j in range(i, self.d + 1):
                At[i, j] = np.prod(np.arange(j - i + 1, j + 1)) * t ** (j - i)
        return At

    def dAT(self, t):
        At = np.zeros((self.q + 1, self.d + 1))
        for i in range(self.q + 1):
            for j in range(i + 1, self.d + 1):
                At[i, j] = (
                    (j - i) * np.prod(np.arange(j - i + 1, j + 1)) * t ** (j - i - 1)
                )
        return At
