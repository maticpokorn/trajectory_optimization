import numpy as np

import osqp
from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import solve
from scipy.sparse import csr_matrix, bmat
import scipy
from constraint_builder import (
    LinearConstraintBuilder,
    GradLinearConstraintBuilder,
    SecondOrderDerivativeLinearConstraintBuilder,
)
from fast_constraint_builder import EqualityConstraintBuilder
from fast_derivative_constraint_builer import (
    EqualityConstraintBuilder1st,
    EqualityConstraintBuilder2nd,
)
from time import time
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csr_matrix, csc_matrix
from scipy.optimize import minimize, LinearConstraint, Bounds
import seaborn as sns


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

    def show_path(self, frame=0):
        if self.dimensions == 2:
            self.show_path_2d()
        if self.dimensions == 3:
            self.show_path_3d(frame)

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

    def show_path_3d(self, frame):
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
        ax.view_init(elev=30, azim=frame)
        plt.colorbar(lc, ax=ax, label="velocity")
        plt.axis("equal")
        # plt.savefig(f"img/frame{frame}.png")
        # plt.close()
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
            """A, b, _ = LinearConstraintBuilder(
                self.waypoints[:, dim], self.d, self.q
            ).build(self.t)"""
            A, b = (
                EqualityConstraintBuilder(self.waypoints[:, dim], self.d, self.q)
                .build(self.t)
                .matrix()
            )
            # self.test_matrices(P, A, b)
            # start = time()
            sol = qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
            # print(f"QP solve time: {time() - start}")
            res_dim = np.array(sol["x"]).reshape((self.n_segments, self.d + 1))
            self.x.append(np.array(sol["x"])[:, 0])
            self.nu.append(np.array(sol["y"])[:, 0])
            result[dim, :, :] = res_dim
            self.result = result
            self.obj += sol["primal objective"]
        return self.obj

    def solve_sparse(self, waypts, t):
        self.waypoints = np.array(waypts)
        self.t = np.array(t)
        self.n_segments = len(t)

        self.obj = 0
        self.nu = []
        self.x = []

        result = np.zeros((self.dimensions, self.n_segments, self.d + 1))
        self.P_sp = self.P_sparse(self.t)
        q = np.zeros(self.n_segments * (self.d + 1))
        for dim in range(self.dimensions):
            A, b = (
                EqualityConstraintBuilder(self.waypoints[:, dim], self.d, self.q)
                .build(self.t)
                .matrix_sparse()
            )
            prob = osqp.OSQP()
            prob.setup(P=self.P_sp, q=q, A=A, l=b, u=b, verbose=False)
            sol = prob.solve()
            res_dim = np.array(sol.x).reshape((self.n_segments, self.d + 1))
            self.x.append(np.array(sol.x))
            self.nu.append(np.array(sol.y))
            result[dim, :, :] = res_dim
            self.result = result
            self.obj += sol.info.obj_val
        return self.obj

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

    def reshape(self, dAT, y, n_segments):
        z = np.zeros(self.d + 1)[:, np.newaxis].T
        print(dAT.shape, y.shape, z.shape)
        print(self.q)
        dAT = np.concatenate((dAT, z)).reshape(n_segments, self.d + 1, self.q + 2)
        y = np.concatenate((y[self.q + 1 :], [0])).reshape(n_segments, self.q + 2)
        return dAT, y

    def build_rhs(self, n_segments, x, y, dPs_, dAs_):
        x = x.reshape(n_segments, self.d + 1)

        # remove reduntant first row (because=0) and add a row so it can be reshaped
        dAs = np.vstack(dAs_[1:] + [np.zeros(self.d + 1)]).reshape(
            n_segments, self.q + 2, self.d + 1
        )
        y = np.hstack((y[self.q + 1 :], [0])).reshape(n_segments, self.q + 2)
        dPx = np.einsum("ijk,ij->ik", np.array(dPs_), x)
        dATy = np.einsum("ijk,ij->ik", dAs, y)
        dAx = np.einsum("ijk,ik->ij", dAs, x)

        b1 = scipy.sparse.block_diag((dPx + dATy)[:, :, np.newaxis], format="csr").T
        mid = scipy.sparse.csr_matrix((n_segments, self.q + 1))
        dAx = dAx[:, :, np.newaxis]
        b2_input = [e for e in dAx[:-1]] + [dAx[-1, :-1]]
        b2 = scipy.sparse.block_diag(b2_input, format="csr").T
        rhs = -scipy.sparse.hstack([b1, mid, b2], format="csr")
        xdPx = np.einsum("bi,bij,bj->b", x, np.array(dPs_), x)
        return rhs, xdPx

    def grad2(self, waypts, t):
        n_segments = len(t)
        grad = np.zeros((self.dimensions, n_segments))
        A, _ = (
            EqualityConstraintBuilder(waypts[:, 0], self.d, self.q)
            .build(t)
            .matrix_sparse()
        )
        dAs = (
            EqualityConstraintBuilder1st(n_segments, self.d, self.q)
            .build(t)
            .constraints
        )
        dPs = self.dPTs(t)
        K = bmat([[self.P_sp, A.T], [A, None]], format="csc")
        lu = splu(K)
        for dim in range(self.dimensions):
            x = self.x[dim]
            y = self.nu[dim]
            rhs, xdPx = self.build_rhs(n_segments, x, y, dPs, dAs)
            dxy = lu.solve(rhs.T.toarray())
            dx = dxy[: (self.d + 1) * n_segments]
            grad[dim] = dx.T @ self.P_sp @ x + 1 / 2 * xdPx
        return np.sum(grad, axis=0)

    def grad(self, waypts, t):
        n_segments = len(t)
        grad = np.zeros((self.dimensions, n_segments))
        grad_ = np.zeros((self.dimensions, n_segments))
        P = self.P(t)
        dPs = self.dP(t)

        A, b = EqualityConstraintBuilder(waypts[:, 0], self.d, self.q).build(t).matrix()
        dAs = (
            EqualityConstraintBuilder1st(n_segments, self.d, self.q).build(t).matrices()
        )
        dAs_ = (
            EqualityConstraintBuilder1st(n_segments, self.d, self.q)
            .build(t)
            .constraints
        )
        dPs_ = self.dPTs(t)

        m, n = A.shape
        K = csc_matrix(np.block([[P, A.T], [A, np.zeros((m, m))]]))
        lu = splu(K)
        # print(f"number of vars: {n}, number of constraints: {m}")
        for dim in range(self.dimensions):
            x = self.x[dim]
            y = self.nu[dim]

            # start = time()
            rhs_, xdPx = self.build_rhs(n_segments, x, y, dPs_, dAs_)
            dxy_ = lu.solve(rhs_.T.toarray())
            dx_ = dxy_[: (self.d + 1) * n_segments]
            # print(dx_.shape, x.shape, P.shape)
            # dxPx = dx_.T @ P @ x
            # print(xdPx.shape, dxPx.shape)
            grad_[dim] = dx_.T @ P @ x + 1 / 2 * xdPx
            # print("vectorized time: ", time() - start)

            start = time()
            for i in range(n_segments):
                dP = dPs[i]
                dA = dAs[i]
                # print("dP")
                # print(dP)
                # print(dP @ x)
                rhs = -np.concatenate((dP @ x + dA.T @ y, dA @ x))
                dxy = lu.solve(rhs)
                # print("error: ", np.mean(np.abs(rhs_[i] - rhs)))
                # print("error dxy: ", np.mean(np.abs(dxy_[:, i] - dxy)))
                dx = dxy[: (self.d + 1) * n_segments][:, np.newaxis]
                dF = dx.T @ P @ x + 1 / 2 * x.T @ dP @ x
                # print("error F: ", np.sum(np.abs(dF - grad_[dim, i])))
                grad[dim, i] = dF
            # print("for loop time: ", time() - start)

        return np.sum(grad, axis=0)

    def build_hess_rhs(self, n_segments, x, y, dx, dy, Ps, dPs, dAs, ddPs, ddAs):

        x = x.reshape(n_segments, self.d + 1)
        dx = dx.reshape(n_segments, self.d + 1, n_segments).transpose(0, 2, 1)

        y = np.hstack((y[self.q + 1 :], [0])).reshape(n_segments, self.q + 2)
        dy = (
            np.vstack((dy[self.q + 1 :], np.zeros((self.n_segments))))
            .reshape(n_segments, self.q + 2, n_segments)
            .transpose(0, 2, 1)
        )

        # remove reduntant first row (because=0) and add a row so it can be reshaped
        dAs = np.vstack(dAs[1:] + [np.zeros(self.d + 1)]).reshape(
            n_segments, self.q + 2, self.d + 1
        )
        ddAs = np.vstack(ddAs[1:] + [np.zeros(self.d + 1)]).reshape(
            n_segments, self.q + 2, self.d + 1
        )
        dP = np.array(dPs)
        ddP = np.array(ddPs)
        ddPx = np.einsum("ijk,ij->ik", ddP, x)
        dPdx = np.einsum("ijk,ilj->ilk", dP, dx)
        ddATy = np.einsum("ijk,ij->ik", ddAs, y)
        dATdy = np.einsum("ijk,ilj->ilk", dAs, dy)
        ddAx = np.einsum("ijk,ik->ij", ddAs, x)
        dAdx = np.einsum("ijk,ilk->ilj", dAs, dx)

        ddPx = np.einsum("ij,ik->ijk", np.eye(n_segments), ddPx)
        ddATy = np.einsum("ij,ik->ijk", np.eye(n_segments), ddATy)
        ddAx = np.einsum("ij,ik->ijk", np.eye(n_segments), ddAx)

        print(x.shape, dPdx.shape, ddPx.shape)
        xdPdx = np.einsum("kj,kij->ki", x, dPdx)
        xddPx = np.einsum("ij,kij->ki", x, ddPx)

        # print(dx.shape, Ps.shape)
        # Pdx = np.einsum("ikj,ilj->ilk", Ps, dx)
        # print(dx.shape, Pdx.shape)
        # dxPdx = np.einsum("ijk,ijl->ij", dx, Pdx)
        # print(dxPdx.shape)
        dxPdx = np.einsum("kil, klm, kjm -> ij", dx, Ps, dx, optimize=True)

        k = np.arange(n_segments**2)
        indices = (k % n_segments) * n_segments + k // n_segments

        dPdx = scipy.sparse.block_diag(dPdx, format="csr")
        dPdxT = dPdx[indices]
        dATdy = scipy.sparse.block_diag(dATdy, format="csr")
        dATdyT = dATdy[indices]
        # ddPx_ddATy = scipy.sparse.block_diag(ddPx + ddATy, format="csr")
        ddPx = scipy.sparse.block_diag(ddPx, format="csr")
        ddATy = scipy.sparse.block_diag(ddATy, format="csr")

        b1 = dPdx + dPdxT + dATdy + dATdyT + ddPx + ddATy

        # dPdx_dATdy = scipy.sparse.block_diag(dPdx + dATdy, format="csr")

        # dPdxT_dATdyT = dPdx_dATdy[indices]
        # b1 = dPdx_dATdy + dPdxT_dATdyT + ddPx_ddATy
        """dPdx_dATdy_ddPx_ddATy = scipy.sparse.block_diag(
            dPdx + dATdy + 0.5 * ddPx + ddATy, format="csr"
        )"""
        # b1 = dPdx_dATdy_ddPx_ddATy + dPdx_dATdy_ddPx_ddATy[indices]

        dAdx = scipy.sparse.block_diag(dAdx, format="csr")
        dAdxT = dAdx[indices]
        ddAx = scipy.sparse.block_diag(ddAx, format="csr")
        b2 = (dAdx + dAdxT + ddAx)[:, :-1]
        # dAdx_ddAx = scipy.sparse.block_diag(dAdx + 0.5 * ddAx, format="csr")
        # b2 = (dAdx_ddAx + dAdx_ddAx[indices])[:, :-1]
        mid = scipy.sparse.csr_matrix((n_segments * n_segments, self.q + 1))
        rhs = -scipy.sparse.hstack([b1, mid, b2], format="csr")

        return rhs, xdPdx, xddPx, dPdx, dxPdx

    def grad_hess2(self, waypts, t):
        n_segments = len(t)
        grad = np.zeros((self.dimensions, n_segments))
        A, b = (
            EqualityConstraintBuilder(waypts[:, 0], self.d, self.q)
            .build(t)
            .matrix_sparse()
        )
        dAs = (
            EqualityConstraintBuilder1st(n_segments, self.d, self.q)
            .build(t)
            .constraints
        )
        ddAs = (
            EqualityConstraintBuilder2nd(n_segments, self.d, self.q)
            .build(t)
            .constraints
        )
        Ps = np.array([self.PT(t) for t in self.t])
        P = self.P_sparse(t)
        dPs = self.dPTs(t)
        ddPs = self.ddPTs(t)

        K = bmat([[P, A.T], [A, None]], format="csc")
        lu = splu(K)
        H = np.zeros((self.dimensions, n_segments, n_segments))
        for dim in range(self.dimensions):
            start = time()
            x = self.x[dim]
            y = self.nu[dim]
            rhs, xdPx = self.build_rhs(n_segments, x, y, dPs, dAs)
            dxy = lu.solve(rhs.T.toarray())
            dx = dxy[: (self.d + 1) * n_segments]
            dy = dxy[(self.d + 1) * n_segments :]
            grad[dim] = dx.T @ self.P_sparse(t) @ x + 1 / 2 * xdPx
            print(f"time for grad: {time() - start}")

            rhs, xdPdx, xddPx, dPdx, dxPdx = self.build_hess_rhs(
                n_segments, x, y, dx, dy, Ps, dPs, dAs, ddPs, ddAs
            )
            rhs = rhs.toarray().T
            # print(K.shape, rhs.shape)
            # print(rhs[0])
            ddxy = lu.solve(rhs).T
            print(ddxy.shape)
            # print(ddxy.shape)
            ddx_ = ddxy[:, : (self.d + 1) * n_segments]
            print(ddx_.shape)
            ddx = ddx_.reshape((n_segments, n_segments, n_segments, self.d + 1))
            x = x.reshape(n_segments, self.d + 1)

            print(x.shape, Ps.shape, ddx.shape)
            xPddx = np.einsum("kl, klm, ijkm -> ij", x, Ps, ddx, optimize=True)

            H[dim] = 1 / 2 * xddPx + xdPdx + xdPdx.T + dxPdx + xPddx

            return rhs.T, xdPdx, xddPx, dPdx, dxPdx, xPddx, ddx_, np.sum(H, axis=0)

    def grad_hess(self, waypts, t):
        start = time()
        rhs_fast, xdPdx, xddPx, dPdx, dxPdx, xPddx, ddx_, H_ = self.grad_hess2(
            waypts, t
        )
        print(f"time for fast hess: {time() - start}")
        n_segments = len(t)
        grad = np.zeros((self.dimensions, n_segments))
        H = np.zeros((self.dimensions, n_segments, n_segments))
        P = self.P(t)
        PTs = self.PTs(t)
        dPs = self.dP(t)
        ddPs = self.ddP(t)

        constr_time = 0
        matmul_time = 0

        start = time()
        A, b = EqualityConstraintBuilder(waypts[:, 0], self.d, self.q).build(t).matrix()
        dAs = (
            EqualityConstraintBuilder1st(n_segments, self.d, self.q).build(t).matrices()
        )
        ddAs = (
            EqualityConstraintBuilder2nd(n_segments, self.d, self.q).build(t).matrices()
        )
        constr_time += time() - start

        m, n = A.shape
        K = csc_matrix(np.block([[P, A.T], [A, np.zeros((m, m))]]))
        lu = splu(K)

        for dim in range(self.dimensions):

            x = self.x[dim]
            y = self.nu[dim]

            dxs = np.zeros((n_segments, (self.d + 1) * n_segments))
            dys = np.zeros((n_segments, m))

            for i in range(n_segments):
                dP = dPs[i]
                dA = dAs[i]
                rhs = -np.concatenate((dP @ x + dA.T @ y, dA @ x))

                dxy = lu.solve(rhs)
                dx = dxy[: (self.d + 1) * n_segments][:, np.newaxis]
                dy = dxy[(self.d + 1) * n_segments :]  # [:, np.newaxis]
                dxs[i] = dx[:, 0]
                dys[i] = dy

                dF = dx.T @ P @ x + 1 / 2 * x.T @ dP @ x
                grad[dim, i] = dF

            # print(PTs.shape, dxs.shape)

            # Pdx = dxs @ PTs

            for i in range(n_segments):
                dAi = dAs[i]
                for j in range(i, n_segments):
                    dAj = dAs[j]

                    start = time()
                    b1 = (
                        dPs[j] @ dxs[i]
                        + dPs[i] @ dxs[j]
                        + dAj.T @ dys[i]
                        + dAi.T @ dys[j]
                    )
                    b2 = dAj @ dxs[i] + dAi @ dxs[j]
                    if i == j:
                        b1 += ddPs[i] @ x + ddAs[i].T @ y
                        b2 += ddAs[i] @ x
                    matmul_time += time() - start

                    rhs = -np.concatenate((b1, b2))
                    # print(i, j, np.mean(np.abs(rhs - rhs_fast[j * n_segments + i])))

                    ddxy = lu.solve(rhs)
                    ddx = ddxy[: (self.d + 1) * n_segments][:, np.newaxis]

                    err = lambda x, y: np.sum(np.abs(x - y))
                    # print(f"dPdx {i} {j}:")
                    # print(dPdx[i * n_segments + j])
                    # print((dPs[i] @ dxs[j]))
                    # print("error:", err(dPdx[i * n_segments + j], dPs[i] @ dxs[j]))
                    # print("dPdxT")
                    # print(dPdxT[i * n_segments + j])
                    # print((dPs[j] @ dxs[i]))
                    # print("error:", err(dPdxT[i * n_segments + j], dPs[j] @ dxs[i]))
                    # print(f"dATdy")
                    # print(dATdy[i * n_segments + j])
                    # print(dAi.T @ dys[j])
                    # print(err(dATdy[i * n_segments + j], dAi.T @ dys[j]))
                    # print(f"dATdyT")
                    # print(dATdyT[i * n_segments + j])
                    # print(dAj.T @ dys[i])
                    # print(err(dATdyT[i * n_segments + j], dAj.T @ dys[i]))
                    # print("ddPx")
                    # print(ddPx[i * n_segments + j])
                    # print(np.sum(ddPx[i * n_segments + j]))
                    # print("ddATy")
                    # print(ddATy[i * n_segments + j])
                    # print(np.sum(ddATy[i * n_segments + j]))
                    # print("b1")
                    # print(b1_fast[i * n_segments + j])
                    # print(b1)
                    # print(err(b1_fast[i * n_segments + j], b1))
                    # print("rhs")
                    # print(rhs_fast[i * n_segments + j])
                    # print(rhs)
                    print(f"dim {dim}, error: {err(rhs_fast[i * n_segments + j], rhs)}")
                    # print(xdPdx[j, i], x.T @ dPs[i] @ dxs[j])
                    print(i, j)
                    print(f"error: {err(xddPx[i, j], x.T @ ddPs[i] @ x)}")
                    # print(xddPx[i, j])
                    print(f"error: {err(xdPdx[j, i], x.T @ dPs[j] @ dxs[i])}")
                    print(f"error: {err(xdPdx[i, j], x.T @ dPs[i] @ dxs[j])}")
                    print(f"error: {err(dxPdx[i,j], dxs[i].T @ P @ dxs[j])}")
                    print(f"error: {err(xPddx[i, j], x.T @ P @ ddx)}")
                    print(f"error: {err(ddx[:,0], ddx_[i * n_segments + j])}")
                    ddF = (
                        dxs[j].T @ P @ dxs[i]
                        + x.T @ dPs[j] @ dxs[i]
                        + x.T @ P @ ddx
                        + x.T @ dPs[i] @ dxs[j]
                    )
                    if i == j:
                        ddF += 1 / 2 * x.T @ ddPs[i] @ x

                    H[dim, i, j] = ddF
                    H[dim, j, i] = ddF

        print(f"H error: {err(H_, H)}")

        print(f"constr time: {constr_time}")
        print(f"matmul time: {matmul_time}")
        return np.sum(grad, axis=0), np.sum(H, axis=0)

    def hess(self, waypts, t):
        n_segments = len(t)
        # grad = np.zeros((self.dimensions, n_segments))
        H = np.zeros((self.dimensions, n_segments, n_segments))
        P = self.P(t)
        dPs = self.dP(t)
        ddPs = self.ddP(t)
        for dim in range(self.dimensions):
            A, b, independent = LinearConstraintBuilder(
                waypts[:, dim], self.d, self.q
            ).build(t)
            dAs = GradLinearConstraintBuilder(waypts[:, dim], self.d, self.q).build(t)
            ddAs = SecondOrderDerivativeLinearConstraintBuilder(
                waypts[:, dim], self.d, self.q
            ).build(t)
            x = self.x[dim]
            y = self.nu[dim]

            m, n = A.shape
            K = csc_matrix(np.block([[P, A.T], [A, np.zeros((m, m))]]))
            lu = splu(K)
            dxs = np.zeros((n_segments, (self.d + 1) * n_segments))
            dys = np.zeros((n_segments, m))
            # print((self.d + 1) * n_segments, m)

            for i in range(n_segments):
                dP = dPs[i]
                dA = dAs[i][independent]
                rhs = -np.concatenate((dP @ x + dA.T @ y, dA @ x))
                # dxy = spsolve(A=K, b=rhs)
                dxy = lu.solve(rhs)
                dx = dxy[: (self.d + 1) * n_segments]  # [:, np.newaxis]
                dy = dxy[(self.d + 1) * n_segments :]  # [:, np.newaxis]
                dxs[i] = dx
                dys[i] = dy
                # print(dx.shape, dy.shape)

            # H = np.zeros((n_segments, n_segments))
            for i in range(n_segments):
                dAi = dAs[i][independent]
                for j in range(i, n_segments):
                    dAj = dAs[j][independent]

                    b1 = (
                        dPs[j] @ dxs[i]
                        + dPs[i] @ dxs[j]
                        + dAj.T @ dys[i]
                        + dAi.T @ dys[j]
                    )
                    b2 = dAj @ dxs[i] + dAi @ dxs[j]
                    if i == j:
                        b1 += ddPs[i] @ x + ddAs[i].T @ y
                        b2 += ddAs[i] @ x

                    rhs = -np.concatenate((b1, b2))
                    # ddxy = spsolve(A=K, b=rhs)
                    ddxy = lu.solve(rhs)

                    # print("accuracy: ", np.mean(K @ ddxy - rhs))
                    ddx = ddxy[: (self.d + 1) * n_segments][:, np.newaxis]
                    # print(ddx.shape)
                    # ddy = ddxy[(self.d + 1) * n_segments :][:, np.newaxis]

                    ddF = (
                        dxs[j].T @ P @ dxs[i]
                        + x.T @ dPs[j] @ dxs[i]
                        + x.T @ P @ ddx
                        + x.T @ dPs[i] @ dxs[j]
                    )
                    if i == j:
                        ddF += 1 / 2 * x.T @ ddPs[i] @ x

                    H[dim, i, j] = ddF
                    H[dim, j, i] = ddF

        return np.sum(H, axis=0)

    def P_sparse(self, t):
        PTs = [self.PT(T) for T in t]
        P = scipy.sparse.block_diag(PTs, format="csc")
        return P

    def P(self, t):
        PTs = [self.PT(T) for T in t]
        P = scipy.linalg.block_diag(
            *PTs
        )  # + 1e-5 * np.eye(self.n_segments * (self.d + 1))
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

    def PTs(self, t):
        PTs = np.concatenate([self.PT(T) for T in t])
        return PTs

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

    def dPTs(self, t):
        return np.array([self.dPT(T) for T in t])

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

    def ddP(self, t):
        ddPs = []
        n = len(t)
        for i in range(n):
            dP = np.zeros((n * (self.d + 1), n * (self.d + 1)))
            dPT = self.ddPT(t[i])
            dP[
                i * (self.d + 1) : (i + 1) * (self.d + 1),
                i * (self.d + 1) : (i + 1) * (self.d + 1),
            ] = dPT
            ddPs.append(dP)
        return np.array(ddPs)

    def ddPTs(self, t):
        return np.array([self.ddPT(T) for T in t])

    def ddPT(self, T):
        ddPT = np.zeros((self.d + 1, self.d + 1))
        for l in range(self.r, self.d + 1):
            for k in range(l, self.d + 1):
                q = (
                    np.prod([(l - m) * (k - m) for m in range(self.r)])
                    * (l + k - 2 * self.r)
                    * T ** (l + k - 2 * self.r - 1)
                )
                ddPT[l, k] = q
                ddPT[k, l] = q
        return ddPT


class NoTimestampSolver(Solver):
    def __init__(self, d, r, q, dimensions, tol=0.01):
        super().__init__(d, r, q, dimensions)
        self.timestamp_solver = TimestampSolver(d, r, q, dimensions)
        self.tol = tol

    def solve(self, waypoints, T, lr=0.3):
        self.waypoints = np.array(waypoints)
        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)
        n = self.waypoints.shape[0] - 1
        # t = np.random.rand(n)
        # t = T / np.sum(t) * t
        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        while dif > self.tol and i < 50:
            # start_time = time()
            obj = self.timestamp_solver.solve_sparse(self.waypoints, t)
            self.intermediate_results.append(obj)
            # print(f"time for qp solve: {time() - start_time}")
            # grad = self.timestamp_solver.grad(self.waypoints, t)
            # start_time = time()
            grad = self.timestamp_solver.grad2(self.waypoints, t)
            # print(f"time for grad: {time() - start_time}")
            # print("error: ", np.sum(np.abs(grad - grad2)))
            # print(grad)
            # print(grad_schur)"""
            grad = grad / np.linalg.norm(grad)
            grad = grad - np.mean(grad)
            t -= lr * grad
            # t *= T / np.sum(t)
            # print(t)
            dif = abs(prev_obj - obj)
            prev_obj = obj
            # self.timestamp_solver.show_path(i)
            i += 1
            print(obj)

        print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj, self.intermediate_results

    def solve_scipy(self, waypoints, T, use_jac=False, use_hess=False):
        self.intermediate_results = []
        self.waypoints = np.array(waypoints)
        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)
        n = self.waypoints.shape[0] - 1
        lc = LinearConstraint(np.ones(n), -np.inf, T)
        bounds = Bounds(np.zeros(n), np.inf)
        options = {"maxiter": 10, "verbose": 3}
        if use_hess:
            res = minimize(
                fun=lambda t: self.timestamp_solver.solve(self.waypoints, t),
                x0=t,
                method="trust-constr",
                jac=lambda t: self.timestamp_solver.grad(self.waypoints, t),
                hess=lambda t: self.timestamp_solver.hess(self.waypoints, t),
                bounds=bounds,
                constraints=[lc],
                options=options,
                callback=self.callback,
            )
        elif use_jac:
            res = minimize(
                fun=lambda t: self.timestamp_solver.solve(self.waypoints, t),
                x0=t,
                method="trust-constr",
                jac=lambda t: self.timestamp_solver.grad(self.waypoints, t),
                bounds=bounds,
                constraints=[lc],
                options=options,
                callback=self.callback,
            )
        else:
            res = minimize(
                fun=lambda t: self.timestamp_solver.solve(self.waypoints, t),
                x0=t,
                method="trust-constr",
                bounds=bounds,
                constraints=[lc],
                options=options,
                callback=self.callback,
            )

        self.obj = res.fun
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return self.obj, self.intermediate_results

    def callback(self, intermediate_result):
        self.intermediate_results.append(intermediate_result.fun)
        print(f"{len(self.intermediate_results)}: {round(intermediate_result.fun, 4)}")

    def solve_newton(self, waypoints, T, lr=1):
        self.waypoints = np.array(waypoints)

        n = self.waypoints.shape[0] - 1
        normal = np.ones(n)
        normal_ = np.ones(n)[:, np.newaxis]
        P = np.eye(n) - (normal_ @ normal_.T) / normal @ normal

        # P = np.eye(n) - np.ones((n, n)) / normal @ normal

        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)

        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        while dif > self.tol and i < 10:
            # start_time = time()
            obj = self.timestamp_solver.solve(self.waypoints, t)
            self.intermediate_results.append(obj)
            # print(f"time for qp solve: {time() - start_time}")
            grad = self.timestamp_solver.grad(self.waypoints, t)
            H = self.timestamp_solver.hess(self.waypoints, t)

            grad_proj = grad - (grad @ normal) / (normal @ normal) * normal

            H_proj = P @ H @ P
            dt = -solve(H_proj, grad_proj)
            # print(grad)
            # print(grad_schur)"""
            # dt = dt / np.linalg.norm(dt)

            # dt_proj = dt - (dt @ normal) / (normal @ normal) * normal

            # v = v / np.linalg.norm(v)
            # v = v - np.mean(v)
            lr = 1
            t_new = t + lr * dt
            t_new = t_new / np.sum(t) * T
            while np.min(t_new) < 0.2:
                lr -= 0.1
                t_new = t + lr * dt
                t_new = t_new / np.sum(t) * T

            t = t_new
            print(lr)
            # t *= T / np.sum(t)
            dif = abs(prev_obj - obj)
            prev_obj = obj
            # self.timestamp_solver.show_path(i)
            i += 1
            print(obj)

        print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj, self.intermediate_results

    def solve_hybrid(self, waypoints, T, k=3, lr=0.3, banded_hessian=False):
        self.waypoints = np.array(waypoints)

        n = self.waypoints.shape[0] - 1
        normal = np.ones(n)
        normal_ = np.ones(n)[:, np.newaxis]
        P = np.eye(n) - (normal_ @ normal_.T) / normal @ normal

        # P = np.eye(n) - np.ones((n, n)) / normal @ normal

        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)

        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        while dif > self.tol and i < 20:
            # start_time = time()
            obj = self.timestamp_solver.solve(self.waypoints, t)
            self.intermediate_results.append(obj)
            # print(f"time for qp solve: {time() - start_time}")

            if i < k:
                grad = self.timestamp_solver.grad(self.waypoints, t)
                grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
                grad = grad / np.linalg.norm(grad)
                grad = grad - np.mean(grad)
                t -= lr * grad
                t = t / np.sum(t) * T
            else:
                grad, H = self.timestamp_solver.grad_hess(self.waypoints, t)
                self.timestamp_solver.grad_hess2(self.waypoints, t)
                if banded_hessian:
                    H = self.banded(H)
                sns.heatmap(
                    H,
                    annot=True,
                    fmt=".2f",
                    cmap="viridis",
                    linewidths=0.5,
                    square=True,
                )

                plt.title("Matrix Heatmap with Annotations")
                plt.show()

                grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
                H_proj = P @ H @ P

                dt = -solve(H_proj, grad_proj)
                t += lr * dt
                t = t / np.sum(t) * T

            dif = abs(prev_obj - obj)
            prev_obj = obj
            i += 1
            print(obj)

        print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj, self.intermediate_results

    def banded(self, M, k=2):
        n, m = M.shape
        i, j = np.ogrid[:n, :m]
        mask = np.abs(i - j) <= k
        return M * mask
