import numpy as np
from scipy.optimize import linprog
import osqp
from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import solve
import sys
import os
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


from scipy.sparse import block_diag, csr_matrix, bmat
import scipy
from .constraint_builder import (
    LinearConstraintBuilder,
    GradLinearConstraintBuilder,
    SecondOrderDerivativeLinearConstraintBuilder,
)
from .fast_constraint_builder import EqualityConstraintBuilder
from .fast_derivative_constraint_builer import (
    EqualityConstraintBuilder1st,
    EqualityConstraintBuilder2nd,
)
from time import time
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csr_matrix, csc_matrix
from scipy.optimize import minimize, LinearConstraint, Bounds
import seaborn as sns
import gurobipy as gp
from gurobipy import GRB


class Trajectory:
    def __init__(self, coeffs, t):
        self.coeffs = coeffs
        self.t = t
        self.dimensions = coeffs.shape[0]
        self.polys = []

        for dim in range(self.dimensions):
            dim_polys = []
            for i in range(coeffs.shape[1]):
                dim_polys.append(np.poly1d(coeffs[dim, i][::-1]))
            self.polys.append(dim_polys)

        # Backward-compatible attributes used elsewhere in the project.
        self.pxs = self.polys[0]
        self.pys = self.polys[1] if self.dimensions > 1 else []
        self.pzs = self.polys[2] if self.dimensions > 2 else []

    def evaluate(self, t):
        # The timestamps t are the durations of each segment.
        # We need to find which segment the time t falls into.
        cumulative_time = 0
        for i in range(len(self.t)):
            segment_duration = self.t[i]
            if t <= cumulative_time + segment_duration:
                # Time within the segment
                time_in_segment = t - cumulative_time
                return np.array(
                    [self.polys[dim][i](time_in_segment) for dim in range(self.dimensions)]
                )
            cumulative_time += segment_duration

        # If t is beyond the total travel time, return the position at the very end.
        return np.array(
            [self.polys[dim][-1](self.t[-1]) for dim in range(self.dimensions)]
        )


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
        _, ax = plt.subplots(figsize=(4, 6))
        # ax.plot(self.waypoints.T[0], self.waypoints.T[1], color="lime")
        # ax.scatter(self.waypoints.T[0], self.waypoints.T[1], color="red")
        """for i, p in enumerate(self.waypoints):
            ax.text(
                p[0],
                p[1],
                f"{np.sum(self.t[:i]):.2f}s ",
                fontsize=9,
                ha="right",
                va="bottom",
            )"""
        x = []
        y = []
        v = []
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            dpx = np.poly1d((coeffs[0, i] * np.arange(self.d + 1))[1:][::-1])
            dpy = np.poly1d((coeffs[1, i] * np.arange(self.d + 1))[1:][::-1])
            T = np.linspace(0, self.t[i], 100)
            x.append(px(T))
            y.append(py(T))
            dx = dpx(T)
            dy = dpy(T)
            v.append(np.sqrt((dx**2 + dy**2))[:-1])

        x = np.hstack(x)
        y = np.hstack(y)
        v = np.hstack(v)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="plasma", array=v, linewidth=2)
        ax.add_collection(lc)
        plt.colorbar(
            lc,
            ax=ax,
            label="Hitrost [m/s]",
            orientation="horizontal",
            pad=0.15,
            shrink=0.95,
            aspect=40.0,
        )
        plt.axis("equal")
        plt.title(f"Optimalna vrednost: {self.obj:.2f}")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
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

    def show_path_2d_latex(self, filename="img/trajectory_2d_velocity.pdf"):
        """Plot 2D trajectory with velocity colormap using LaTeX fonts and export to PDF"""
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        coeffs = self.result
        fig, ax = plt.subplots(figsize=(8, 8))

        x = []
        y = []
        v = []
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            dpx = np.poly1d((coeffs[0, i] * np.arange(self.d + 1))[1:][::-1])
            dpy = np.poly1d((coeffs[1, i] * np.arange(self.d + 1))[1:][::-1])
            T = np.linspace(0, self.t[i], 100)
            x.append(px(T))
            y.append(py(T))
            dx = dpx(T)
            dy = dpy(T)
            v.append(np.sqrt((dx**2 + dy**2))[:-1])

        x = np.hstack(x)
        y = np.hstack(y)
        v = np.hstack(v)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="plasma", array=v, linewidth=2)
        ax.add_collection(lc)

        # Add waypoints with cumulative time
        waypoints_array = np.array(self.waypoints)
        ax.scatter(
            waypoints_array[:, 0],
            waypoints_array[:, 1],
            facecolors="none",
            s=50,
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
        )
        for i, p in enumerate(self.waypoints):
            cumulative_time = np.sum(self.t[:i])
            # Calculate offset based on data range
            x_range = max(waypoints_array[:, 0]) - min(waypoints_array[:, 0])
            y_range = max(waypoints_array[:, 1]) - min(waypoints_array[:, 1])
            offset = max(x_range, y_range) * 0.03
            ax.text(
                p[0] + offset,
                p[1] + offset,
                f"${cumulative_time:.2f}$ s",
                fontsize=12,
                ha="left",
                va="bottom",
            )

        ax.autoscale()
        # Add padding to axis limits and make square
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_margin = (xlim[1] - xlim[0]) * 0.1
        y_margin = (ylim[1] - ylim[0]) * 0.1
        x_range = (xlim[1] - xlim[0]) + 2 * x_margin
        y_range = (ylim[1] - ylim[0]) + 2 * y_margin

        # Use the larger range for both axes to create a square
        max_range = max(x_range, y_range)
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
        ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
        cbar = plt.colorbar(
            lc,
            ax=ax,
            label=r"Hitrost [m/s]",
            orientation="horizontal",
            pad=0.15,
            shrink=0.7,
            aspect=60.0,
        )
        cbar.set_label(r"Hitrost [m/s]", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        ax.set_aspect("equal")
        ax.set_title(f"Optimalna vrednost: {self.obj:.2f}", fontsize=18)
        ax.set_xlabel(r"$x$ [m]", fontsize=16)
        ax.set_ylabel(r"$y$ [m]", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.show()

    def show_path_3d_latex(self, frame=0, filename="img/trajectory_3d_velocity.pdf"):
        """Plot 3D trajectory from three different viewing angles using LaTeX fonts and export to PDF"""
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        coeffs = self.result
        fig = plt.figure(figsize=(18, 5))

        # Compute trajectory first
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

        # Define three different viewing angles (elevation, azimuth)
        viewing_angles = [
            (30, 45),  # View 1: Standard isometric view
            (20, 135),  # View 2: From the back-side
            (40, -45),  # View 3: From the opposite side
        ]

        # Set limits based on actual data with padding
        x_margin = (x.max() - x.min()) * 0.1
        y_margin = (y.max() - y.min()) * 0.1
        z_margin = (z.max() - z.min()) * 0.1
        xlim = [x.min() - x_margin, x.max() + x_margin]
        ylim = [y.min() - y_margin, y.max() + y_margin]
        zlim = [z.min() - z_margin, z.max() + z_margin]

        for idx, (elev, azim) in enumerate(viewing_angles):
            ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

            # Plot waypoints with annotations
            ax.scatter(
                self.waypoints.T[0],
                self.waypoints.T[1],
                self.waypoints.T[2],
                facecolors="none",
                s=60,
                zorder=5,
                edgecolors="black",
                linewidths=1.5,
            )

            # Add time annotations to waypoints
            for i, p in enumerate(self.waypoints):
                cumulative_time = np.sum(self.t[:i])
                x_range = x.max() - x.min()
                y_range = y.max() - y.min()
                z_range = z.max() - z.min()
                offset = max(x_range, y_range, z_range) * 0.03
                ax.text(
                    p[0] + offset,
                    p[1] + offset,
                    p[2] + offset,
                    f"${cumulative_time:.2f}$ s",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                )

            # Create 3D line segments colored by velocity
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap="plasma", array=v, linewidth=2.5)
            ax.add_collection(lc)

            # Set consistent limits for all subplots
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            # Set viewing angle
            ax.view_init(elev=elev, azim=azim)

            # Add colorbar only to the last subplot
            if idx == 2:
                cbar = plt.colorbar(
                    lc, ax=ax, label=r"Hitrost [m/s]", pad=0.1, shrink=0.8
                )
                cbar.set_label(r"Hitrost [m/s]", fontsize=14)
                cbar.ax.tick_params(labelsize=11)

            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel(r"$x$ [m]", fontsize=12)
            ax.set_ylabel(r"$y$ [m]", fontsize=12)
            ax.set_zlabel(r"$z$ [m]", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Optimalna vrednost $F$: {self.obj:.2f}", fontsize=18, y=0.98)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.show()

    def show_path_colored_2d(self):
        """Plot 2D trajectory with distinct colors for each segment and waypoint"""
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        coeffs = self.result
        _, ax = plt.subplots(figsize=(8, 8))

        # Define colors for segments (RGB values converted to 0-1 scale)
        segment_colors = [
            (51 / 255, 153 / 255, 255 / 255),  # Block 1: Bright Blue
            (255 / 255, 51 / 255, 255 / 255),  # Block 2: Bright Magenta
            (255 / 255, 51 / 255, 51 / 255),  # Block 3: Bright Red
            (51 / 255, 204 / 255, 51 / 255),  # Block 4: Bright Green
            (51 / 255, 204 / 255, 51 / 255),  # Block 5: Bright Green
            (255 / 255, 153 / 255, 51 / 255),  # Block 6: Bright Orange
        ]

        # Define colors for waypoints
        waypoint_colors = [
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "navy",
            "teal",
        ]

        # Plot each segment with its own color
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            T = np.linspace(0, self.t[i], 100)
            x = px(T)
            y = py(T)
            color = segment_colors[i % len(segment_colors)]
            ax.plot(x, y, color=color, linewidth=2, label=f"Segment {i+1}")

            # Add segment time annotation at the middle of the segment
            mid_x = px((2 - i) * self.t[i] / 3)
            mid_y = py((2 - i) * self.t[i] / 3)
            ax.text(
                mid_x,
                mid_y,
                f"${self.t[i]:.2f}$ s",
                fontsize=20,
                ha="left",
                va="center",
            )

        # Plot waypoints with distinct colors
        for i, p in enumerate(self.waypoints):
            color = waypoint_colors[i % len(waypoint_colors)]
            ax.scatter(
                p[0],
                p[1],
                color=color,
                s=100,
                zorder=5,
            )
            ax.text(p[0], p[1], f"  $P_{{{i}}}$", fontsize=22, ha="left", va="bottom")

        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m]", fontsize=20)
        ax.set_ylabel(r"$y$ [m]", fontsize=20)
        ax.set_title(f"Optimalna vrednost: {self.obj:.2f}", fontsize=24)
        ax.legend(fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        plt.tight_layout()
        plt.savefig("img/trajectory_2d.pdf", bbox_inches="tight")
        plt.show()

    def show_path_colored_3d(self, frame=0):
        """Plot 3D trajectory with distinct colors for each segment and waypoint"""
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        coeffs = self.result
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Define colors for segments
        segment_colors = [
            "blue",
            "magenta",
            "cyan",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "navy",
        ]

        # Define colors for waypoints
        waypoint_colors = [
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "navy",
            "teal",
        ]

        # Plot each segment with its own color
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            pz = np.poly1d(coeffs[2, i][::-1])
            T = np.linspace(0, self.t[i], 100)
            x = px(T)
            y = py(T)
            z = pz(T)
            color = segment_colors[i % len(segment_colors)]
            ax.plot(x, y, z, color=color, linewidth=2, label=f"Segment {i+1}")

        # Plot waypoints with distinct colors
        for i, p in enumerate(self.waypoints):
            color = waypoint_colors[i % len(waypoint_colors)]
            ax.scatter(
                p[0],
                p[1],
                p[2],
                color=color,
                s=100,
                zorder=5,
            )
            ax.text(
                p[0],
                p[1],
                p[2],
                f"  $P_{{{i}}}$",
                fontsize=14,
                ha="left",
                va="bottom",
            )

        ax.grid(True, alpha=0.3)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel(r"$x$ [m]", fontsize=12)
        ax.set_ylabel(r"$y$ [m]", fontsize=12)
        ax.set_zlabel(r"$z$ [m]", fontsize=12)
        ax.set_title(f"Optimalna vrednost: {self.obj:.2f}", fontsize=18)
        ax.view_init(elev=30, azim=frame)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig("img/trajectory_3d.pdf", bbox_inches="tight")
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
        try:
            self.waypoints = np.array(waypts)
            self.t = np.array(t)
            self.n_segments = len(t)
            self.obj = 0
            self.nu = []
            self.x = []

            result = np.zeros((self.dimensions, self.n_segments, self.d + 1))
            self.P_sp = self.P_sparse(self.t)
            # print("P:", self.P_sp.toarray())
            q = np.zeros(self.n_segments * (self.d + 1))

            for dim in range(self.dimensions):
                # Build constraints
                A, b = (
                    EqualityConstraintBuilder(self.waypoints[:, dim], self.d, self.q)
                    .build(self.t)
                    .matrix_sparse()
                )
                # print(f"A, b (dim = {dim}):", A.toarray(), b)

                prob = osqp.OSQP()
                with suppress_stderr():
                    prob.setup(P=self.P_sp, q=q, A=A, l=b, u=b, verbose=False)
                    sol = prob.solve()

                res_dim = np.array(sol.x).reshape((self.n_segments, self.d + 1))

                self.x.append(np.array(sol.x))
                self.nu.append(np.array(sol.y))
                result[dim, :, :] = res_dim
                self.obj += sol.info.obj_val

            if self.obj < 0.1:
                return np.inf
            self.result = result
            return self.obj
        except Exception as e:
            print(f"Trajectory Optimization failed: {e}")
            return np.inf

    def solve_gurobi(self, waypts, t):
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
            m = gp.Model("Trajectory_QP")
            # Suppress all console output
            m.Params.OutputFlag = 0
            A, b = (
                EqualityConstraintBuilder(self.waypoints[:, dim], self.d, self.q)
                .build(self.t)
                .matrix_sparse()
            )
            x = m.addMVar(
                shape=self.n_segments * (self.d + 1), lb=-float("inf"), name="coeffs"
            )

            m.setObjective(0.5 * x @ self.P_sp @ x, GRB.MINIMIZE)

            con = m.addConstr(A @ x == b, name="continuity_and_waypoints")

            m.optimize()
            if m.status == GRB.OPTIMAL:
                res_dim = np.array(x.X).reshape((self.n_segments, self.d + 1))
                self.x.append(np.array(x.X))
                self.nu.append(np.array(con.Pi))
                result[dim, :, :] = res_dim
                self.result = result
                self.obj += m.ObjVal
            else:
                print("solve failed")
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
                # print(f"dxy (dim={dim}, segment={i}): {dxy}")
                # print("error: ", np.mean(np.abs(rhs_[i] - rhs)))
                # print("error dxy: ", np.mean(np.abs(dxy_[:, i] - dxy)))
                dx = dxy[: (self.d + 1) * n_segments][:, np.newaxis]
                dF = dx.T @ P @ x + 1 / 2 * x.T @ dP @ x
                print(f"grad (dim={dim}, segment={i}): {dF}")
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
        xdPdx = np.einsum("kj,kij->ki", x, dPdx)
        xddPx = np.einsum("ij,kij->ki", x, ddPx)

        dxPdx = np.einsum("kil, klm, kjm -> ij", dx, Ps, dx, optimize=True)

        k = np.arange(n_segments**2)
        indices = (k % n_segments) * n_segments + k // n_segments

        dPdx = scipy.sparse.block_diag(dPdx, format="csr")
        dPdxT = dPdx[indices]
        dATdy = scipy.sparse.block_diag(dATdy, format="csr")
        dATdyT = dATdy[indices]
        ddPx = scipy.sparse.block_diag(ddPx, format="csr")
        ddATy = scipy.sparse.block_diag(ddATy, format="csr")

        b1 = dPdx + dPdxT + dATdy + dATdyT + ddPx + ddATy

        dAdx = scipy.sparse.block_diag(dAdx, format="csr")
        dAdxT = dAdx[indices]
        ddAx = scipy.sparse.block_diag(ddAx, format="csr")
        b2 = (dAdx + dAdxT + ddAx)[:, :-1]
        mid = scipy.sparse.csr_matrix((n_segments * n_segments, self.q + 1))
        rhs = -scipy.sparse.hstack([b1, mid, b2], format="csr")

        return rhs, xdPdx, xddPx, dxPdx

    def banded_indices(self, bands=3):
        # rhs is of shape n_segments * n_segments (1 for each entry of Hessian)
        # we want only to solve for banded elements
        # H[i,j] <-> rhs[n_segments * i + j]
        Is = []
        Js = []
        Is.append(np.arange(self.n_segments))
        Js.append(np.arange(self.n_segments))
        for k in range(1, bands):
            # lower band
            Is.append(np.arange(k, self.n_segments))
            Js.append(np.arange(self.n_segments - k))
            # upper band
            Is.append(np.arange(self.n_segments - k))
            Js.append(np.arange(k, self.n_segments))
        return np.concatenate(Is), np.concatenate(Js)

    def grad_hess2(self, waypts, t, bands=None):
        n_segments = len(t)
        H = np.zeros((self.dimensions, n_segments, n_segments))
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
        ddAs = (
            EqualityConstraintBuilder2nd(n_segments, self.d, self.q)
            .build(t)
            .constraints
        )
        Ps = np.array([self.PT(t) for t in self.t])
        dPs = self.dPTs(t)
        ddPs = self.ddPTs(t)

        K = bmat([[self.P_sp, A.T], [A, None]], format="csc")
        lu = splu(K)

        for dim in range(self.dimensions):
            x = self.x[dim]
            y = self.nu[dim]
            rhs, xdPx = self.build_rhs(n_segments, x, y, dPs, dAs)
            dxy = lu.solve(rhs.T.toarray())
            dx = dxy[: (self.d + 1) * n_segments]
            dy = dxy[(self.d + 1) * n_segments :]
            grad[dim] = dx.T @ self.P_sp @ x + 1 / 2 * xdPx

            rhs, xdPdx, xddPx, dxPdx = self.build_hess_rhs(
                n_segments, x, y, dx, dy, Ps, dPs, dAs, ddPs, ddAs
            )
            # start = time()
            if bands:
                ddx = np.zeros((n_segments * n_segments, n_segments * (self.d + 1)))
                Is, Js = self.banded_indices(bands)
                rhs = rhs[Is * n_segments + Js].toarray().T
                # start = time()
                ddxy = lu.solve(rhs).T
                # print(f"banded system solve time: {time() - start}")
                ddx[Is * n_segments + Js] = ddxy[:, : (self.d + 1) * n_segments]
            else:
                rhs = rhs.toarray().T
                ddxy = lu.solve(rhs).T
                ddx = ddxy[:, : (self.d + 1) * n_segments]

            # print(f"system solve time: {time() - start}")

            ddx = ddx.reshape((n_segments, n_segments, n_segments, self.d + 1))
            x = x.reshape(n_segments, self.d + 1)

            xPddx = np.einsum("kl,klm,ijkm->ij", x, Ps, ddx, optimize=True)

            H[dim] = 1 / 2 * xddPx + xdPdx + xdPdx.T + dxPdx + xPddx

        H_total = np.sum(H, axis=0)
        if bands is not None:
            eigenvalues = np.linalg.eigvalsh(H_total)
            min_eig = eigenvalues[0]
            if np.abs(min_eig) < 1e-10:
                print(f"Banded Hessian (bands={bands}): singular (min eigenvalue ≈ 0)")
            elif min_eig > 0:
                print(f"Banded Hessian (bands={bands}): positive definite (min eigenvalue = {min_eig:.4e})")
            else:
                print(f"Banded Hessian (bands={bands}): NOT positive definite (min eigenvalue = {min_eig:.4e})")
        return np.sum(grad, axis=0), H_total

    def grad_hess(self, waypts, t):
        start = time()
        grad_, H_ = self.grad_hess2(waypts, t)
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
                dxs[i] = dx
                dys[i] = dy

                dF = dx.T @ P @ x + 1 / 2 * x.T @ dP @ x
                grad[dim, i] = dF

            # print(PTs.shape, dxs.shape)

            # Pdx = dxs @ PTs

            for i in range(n_segments):
                dAi = dAs[i]
                for j in range(i, n_segments):
                    dAj = dAs[j]

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

        # print(f"H error: {err(H_, H)}")

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

    def eval(self, t):
        y = self.timestamp_solver.solve_sparse(self.waypoints, t)
        if y == np.inf:
            return np.inf
        self.J, self.H = self.timestamp_solver.grad_hess2(self.waypoints, t, bands=4)
        return y

    def solve_trust_constr(self, waypoints, T):
        self.intermediate_results = []
        self.intermediate_times = []
        self.start_time = time()
        self.waypoints = np.array(waypoints)
        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)
        n = self.waypoints.shape[0] - 1
        lc = LinearConstraint(np.ones(n), -np.inf, T)
        bounds = Bounds(np.zeros(n), np.inf)
        options = {"maxiter": 20, "verbose": 0, "xtol": self.tol, "gtol": 1e-1}
        res = minimize(
            fun=lambda t: self.eval(t),
            x0=t,
            method="trust-constr",
            jac=lambda t: self.J,
            hess=lambda t: self.H,
            bounds=bounds,
            constraints=[lc],
            options=options,
            callback=self.callback,
        )
        self.obj = res.fun
        self.x = self.timestamp_solver.x
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return self.obj, self.intermediate_results, self.intermediate_times

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
            obj = self.timestamp_solver.solve_gurobi(self.waypoints, t)
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
            # print(obj)

        # print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj, self.intermediate_results

    def solve_scipy(self, waypoints, T, use_jac=False, use_hess=False):
        self.intermediate_results = []
        self.intermediate_times = []
        self.start_time = time()
        self.waypoints = np.array(waypoints)
        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)
        n = self.waypoints.shape[0] - 1
        lc = LinearConstraint(np.ones(n), -np.inf, T)
        bounds = Bounds(np.zeros(n), np.inf)
        options = {"maxiter": 15, "verbose": 3, "xtol": self.tol}
        if use_hess:
            res = minimize(
                fun=lambda t: self.eval(t),
                x0=t,
                method="trust-constr",
                jac=lambda t: self.J,
                hess=lambda t: self.H,
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
        return self.obj, self.intermediate_results, self.intermediate_times

    def callback(self, intermediate_result):
        self.intermediate_results.append(intermediate_result.fun)
        self.intermediate_times.append(time() - self.start_time)
        # print(f"{len(self.intermediate_results)}: {round(intermediate_result.fun, 4)}")

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
            # print(lr)
            # t *= T / np.sum(t)
            dif = abs(prev_obj - obj)
            prev_obj = obj
            # self.timestamp_solver.show_path(i)
            i += 1
            # print(obj)

        # print(f"iter: {i}")
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
            # print(f"min t: {np.min(t)}")
            # start_time = time()
            obj = self.timestamp_solver.solve_sparse(self.waypoints, t)
            self.intermediate_results.append(obj)
            # print(f"time for qp solve: {time() - start_time}")

            if i < k:
                grad = self.timestamp_solver.grad2(self.waypoints, t)
                grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
                grad = grad / np.linalg.norm(grad)
                grad = grad - np.mean(grad)
                t -= lr * grad
                t = t / np.sum(t) * T
            else:
                # grad2 = self.timestamp_solver.grad2(self.waypoints, t)
                grad, H = self.timestamp_solver.grad_hess2(self.waypoints, t, bands=4)
                """grad_, H_ = self.timestamp_solver.grad_hess(self.waypoints, t)
                print(
                    f"error: grad: {np.median(np.abs(grad - grad_))}, hess: {np.mean(np.abs(H - H_))}"
                )
                print(f"error: grad: {np.mean(np.abs(grad - grad2))}")"""
                # self.timestamp_solver.grad_hess2(self.waypoints, t)
                if banded_hessian:
                    H = self.banded(H)
                """sns.heatmap(
                    H,
                    annot=False,
                    fmt=".2f",
                    cmap="viridis",
                    linewidths=0.5,
                    square=True,
                )"""
                """plt.imshow(H)
                plt.colorbar()

                plt.title("Hessian matrix heatmap")
                plt.show()"""

                grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
                H_proj = P @ H @ P

                dt = -solve(H_proj, grad_proj)
                t += lr * dt
                t = t / np.sum(t) * T

            dif = abs(prev_obj - obj)
            prev_obj = obj
            i += 1
            # print(obj)

        # print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj, self.intermediate_results

    def solve_backtracking_line_search(
        self, waypoints, T, tau=0.5, c=0.5, bands=None, intermediate_traj=False, x0=None
    ):

        self.waypoints = np.array(waypoints)

        n = self.waypoints.shape[0] - 1
        normal = np.ones(n)
        normal_ = np.ones(n)[:, np.newaxis]
        P = np.eye(n) - (normal_ @ normal_.T) / (normal @ normal)

        if x0 is not None:
            t = np.array(x0)
        else:
            dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
            t = T * dists / np.sum(dists)

        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        self.intermediate_times = []
        intermediate_trajs = []
        start_time = time()
        while dif > self.tol and i < 50:
            obj = self.timestamp_solver.solve_sparse(self.waypoints, t)
            self.intermediate_results.append(obj)
            self.intermediate_times.append(time() - start_time)

            grad, H = self.timestamp_solver.grad_hess2(self.waypoints, t, bands=bands)
            np.save("hessian.npy", H)
            """plt.imshow(H)
            plt.colorbar()
            plt.savefig("img/hessian.pdf")
            plt.show()"""
            grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
            # print(f"projected gradient at t={t}: {grad_proj}")
            # grad = grad / np.linalg.norm(grad)
            # dt_grad = grad - np.mean(grad)
            dt_grad = grad_proj / np.linalg.norm(grad_proj)
            # print("normalized projected gradient: ", dt_grad)
            m_grad = dt_grad @ dt_grad

            # H_proj = P @ H @ P

            K = bmat([[H, normal_], [normal_.T, np.zeros((1, 1))]], format="csr")
            rhs = np.concatenate([-grad, [0]])
            sol = solve(K.toarray(), rhs)
            dt_hess = sol[:-1]
            m_hess = dt_hess @ grad_proj

            alpha = 1.0
            for _ in range(50):
                t_grad = t - alpha * dt_grad
                t_grad = t_grad / np.sum(t_grad) * T
                obj_grad = self.timestamp_solver.solve_sparse(self.waypoints, t_grad)
                traj_grad = Trajectory(
                    self.timestamp_solver.result, self.timestamp_solver.t
                )

                t_hess = t + alpha * dt_hess
                t_hess = t_hess / np.sum(t_hess) * T
                obj_hess = self.timestamp_solver.solve_sparse(self.waypoints, t_hess)
                traj_hess = Trajectory(
                    self.timestamp_solver.result, self.timestamp_solver.t
                )
                # print(t_hess, obj_hess)
                # print(t_grad, obj_grad)
                if (
                    obj_grad < obj + c * alpha * m_grad
                    or obj_hess < obj + c * alpha * m_hess
                ):
                    if obj_hess < obj_grad or np.isnan(obj_grad):
                        t = t_hess
                        obj = obj_hess
                        """print(
                            f"used hess with alpha {alpha} by {obj_grad - obj_hess:<.4f}"
                        )"""
                        if intermediate_traj:
                            intermediate_trajs.append(traj_hess)
                    else:
                        t = t_grad
                        obj = obj_grad
                        """print(
                            f"used grad with alpha {alpha} by {obj_hess - obj_grad:<.4f}"
                        )"""
                        if intermediate_traj:
                            intermediate_trajs.append(traj_grad)

                    break

                alpha *= tau

            dif = abs(prev_obj - obj)
            prev_obj = obj

            i += 1
            # print(obj)

        # print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return (
            obj,
            self.intermediate_results,
            intermediate_trajs,
            self.intermediate_times,
        )

    def solve_backtracking_line_search_hess(
        self, waypoints, T, tau=0.5, c=0.5, bands=None, intermediate_traj=False, x0=None
    ):

        self.waypoints = np.array(waypoints)

        n = self.waypoints.shape[0] - 1
        normal = np.ones(n)
        normal_ = np.ones(n)[:, np.newaxis]

        if x0 is not None:
            t = np.array(x0)
        else:
            dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
            t = T * dists / np.sum(dists)

        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        self.intermediate_times = []
        intermediate_trajs = []
        start_time = time()
        while dif > self.tol and i < 50:
            obj = self.timestamp_solver.solve_sparse(self.waypoints, t)
            self.intermediate_results.append(obj)
            self.intermediate_times.append(time() - start_time)

            grad, H = self.timestamp_solver.grad_hess2(self.waypoints, t, bands=bands)
            grad_proj = grad - (grad @ normal) / (normal @ normal) * normal

            K = bmat([[H, normal_], [normal_.T, np.zeros((1, 1))]], format="csr")
            rhs = np.concatenate([-grad, [0]])
            sol = solve(K.toarray(), rhs)
            dt_hess = sol[:-1]
            m_hess = dt_hess @ grad_proj

            alpha = 1.0
            for _ in range(50):

                t_hess = t + alpha * dt_hess
                t_hess = t_hess / np.sum(t_hess) * T
                obj_hess = self.timestamp_solver.solve_sparse(self.waypoints, t_hess)
                traj_hess = Trajectory(
                    self.timestamp_solver.result, self.timestamp_solver.t
                )
                if obj_hess < obj + c * alpha * m_hess:
                    t = t_hess
                    obj = obj_hess
                    """print(
                        f"used hess with alpha {alpha} by {obj_grad - obj_hess:<.4f}"
                    )"""
                    if intermediate_traj:
                        intermediate_trajs.append(traj_hess)
                    break

                alpha *= tau

            dif = abs(prev_obj - obj)
            prev_obj = obj

            i += 1
            # print(obj)

        # print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return (
            obj,
            self.intermediate_results,
            intermediate_trajs,
            self.intermediate_times,
        )

    def solve_backtracking_line_search_grad(self, waypoints, T, tau=0.5, c=0.0):

        self.waypoints = np.array(waypoints)

        n = self.waypoints.shape[0] - 1
        normal = np.ones(n)

        dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
        t = T * dists / np.sum(dists)

        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        self.intermediate_times = []
        start_time = time()
        while dif > self.tol and i < 50:
            obj = self.timestamp_solver.solve_sparse(self.waypoints, t)
            self.intermediate_results.append(obj)
            self.intermediate_times.append(time() - start_time)

            grad = self.timestamp_solver.grad2(self.waypoints, t)
            grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
            grad = grad / np.linalg.norm(grad)
            # dt_grad = grad - np.mean(grad)
            dt_grad = grad_proj / np.linalg.norm(grad_proj)
            # print(f"projected gradient {dt_grad}")
            m_grad = dt_grad @ dt_grad

            alpha = 1.0
            for _ in range(20):
                t_grad = t - alpha * dt_grad
                t_grad = t_grad / np.sum(t_grad) * T
                obj_grad = self.timestamp_solver.solve_sparse(self.waypoints, t_grad)

                if obj_grad < obj + c * alpha * m_grad:
                    t = t_grad
                    obj = obj_grad
                    break

                alpha *= tau

            dif = abs(prev_obj - obj)
            prev_obj = obj
            i += 1
            # print(obj)

        # print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return obj, self.intermediate_results, self.intermediate_times

    def solve_backtracking_line_search2(
        self, waypoints, T, tau=0.5, c=0.5, bands=None, intermediate_traj=False, x0=None
    ):

        self.waypoints = np.array(waypoints)

        n = self.waypoints.shape[0] - 1
        normal = np.ones(n)
        normal_ = np.ones(n)[:, np.newaxis]
        P = np.eye(n) - (normal_ @ normal_.T) / (normal @ normal)

        if x0 is not None:
            t = np.array(x0)
        else:
            dists = np.linalg.norm((self.waypoints[1:] - self.waypoints[:-1]), axis=1)
            t = T * dists / np.sum(dists)

        prev_obj = 1e12
        dif = 1e12
        i = 0
        self.intermediate_results = []
        self.intermediate_times = []
        intermediate_trajs = []
        start_time = time()
        while dif > self.tol and i < 50:
            obj = self.timestamp_solver.solve_sparse(self.waypoints, t)
            self.intermediate_results.append(obj)
            self.intermediate_times.append(time() - start_time)

            grad, H = self.timestamp_solver.grad_hess2(self.waypoints, t, bands=bands)
            """plt.imshow(H)
            plt.colorbar()
            plt.savefig("img/hessian.pdf")
            plt.show()"""
            grad_proj = grad - (grad @ normal) / (normal @ normal) * normal
            # print(f"projected gradient at t={t}: {grad_proj}")
            # grad = grad / np.linalg.norm(grad)
            # dt_grad = grad - np.mean(grad)
            dt_grad = grad_proj / np.linalg.norm(grad_proj)
            # print("normalized projected gradient: ", dt_grad)
            m_grad = dt_grad @ dt_grad

            # H_proj = P @ H @ P

            K = bmat([[H, normal_], [normal_.T, np.zeros((1, 1))]], format="csr")
            rhs = np.concatenate([-grad, [0]])
            sol = solve(K.toarray(), rhs)
            dt_hess = sol[:-1]
            m_hess = dt_hess @ grad_proj

            alpha = 1.0
            for _ in range(50):
                t_hess = t + alpha * dt_hess
                t_hess = t_hess / np.sum(t_hess) * T
                obj_hess = self.timestamp_solver.solve_sparse(self.waypoints, t_hess)

                if obj_hess < obj + c * alpha * m_hess:
                    t = t_hess
                    obj = obj_hess
                    break

                # Only evaluate gradient step if Newton failed
                t_grad = t - alpha * dt_grad
                t_grad = t_grad / np.sum(t_grad) * T
                obj_grad = self.timestamp_solver.solve_sparse(self.waypoints, t_grad)

                if obj_grad < obj + c * alpha * m_grad:
                    t = t_grad
                    obj = obj_grad
                    break

                alpha *= tau

            dif = abs(prev_obj - obj)
            prev_obj = obj

            i += 1
            # print(obj)

        # print(f"iter: {i}")
        self.obj = obj
        self.x = self.timestamp_solver.obj
        self.result = self.timestamp_solver.result
        self.t = self.timestamp_solver.t
        return (
            obj,
            self.intermediate_results,
            intermediate_trajs,
            self.intermediate_times,
        )


def banded(M, k=2):
    n, m = M.shape
    i, j = np.ogrid[:n, :m]
    mask = np.abs(i - j) < k
    return M * mask
