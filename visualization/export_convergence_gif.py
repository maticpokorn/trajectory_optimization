"""
Export a GIF visualizing 3D trajectory optimization convergence.
Each convergence step is shown with a smoothly rotating (spinning) camera.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from source.solver import NoTimestampSolver

# ── Configuration ────────────────────────────────────────────────────────────
WAYPOINTS = np.array([
    [0, 0, 20],
    [0,   0,   0],
    [8,   4,  12],
    [20, 14,  18],
    [8,  22,  -4],
    [3,  20,  14],
    [0, 18, 16],
    [15, 0, -5],
    [10, 10, -5],
    [25, 25, 25]
], dtype=float)
T_TOTAL          = 10.0   # total travel time budget
D                = 9      # polynomial degree
R                = 4      # derivative order to minimize
Q                = 4      # continuity order

OUTPUT_FILE      = "img/convergence.gif"
FPS              = 12     # frames per second in GIF
DPI              = 120    # render resolution
FRAMES_PER_STEP  = 2      # camera frames rendered per convergence step (fast)
CAMERA_DEG_STEP  = 1.0    # degrees the camera rotates per frame (slow spin)
CAMERA_ELEV      = 25     # camera elevation angle (degrees)
GHOST_STEPS      = 4      # how many previous iterations to show as ghost trails
GHOST_ALPHA_MIN  = 0.20   # opacity of the oldest ghost
GHOST_ALPHA_MAX  = 0.55   # opacity of the most-recent ghost
# ─────────────────────────────────────────────────────────────────────────────


def trajectory_xyz_speed(traj, d):
    """Return (x, y, z, speed) arrays from a Trajectory object."""
    coeffs = traj.coeffs  # (3, n_segments, d+1)
    t_segs = traj.t
    x_all, y_all, z_all, v_all = [], [], [], []
    for i in range(coeffs.shape[1]):
        px  = np.poly1d(coeffs[0, i][::-1])
        py  = np.poly1d(coeffs[1, i][::-1])
        pz  = np.poly1d(coeffs[2, i][::-1])
        dpx = np.poly1d((coeffs[0, i] * np.arange(d + 1))[1:][::-1])
        dpy = np.poly1d((coeffs[1, i] * np.arange(d + 1))[1:][::-1])
        dpz = np.poly1d((coeffs[2, i] * np.arange(d + 1))[1:][::-1])
        T   = np.linspace(0, t_segs[i], 100)
        x_all.append(px(T)); y_all.append(py(T)); z_all.append(pz(T))
        dx = dpx(T); dy = dpy(T); dz = dpz(T)
        v_all.append(np.sqrt(dx**2 + dy**2 + dz**2)[:-1])
    return (np.hstack(x_all), np.hstack(y_all),
            np.hstack(z_all), np.hstack(v_all))


def compute_global_limits(trajs, d):
    """Axis + colormap limits consistent across all frames."""
    all_x, all_y, all_z, all_v = [], [], [], []
    for traj in trajs:
        x, y, z, v = trajectory_xyz_speed(traj, d)
        all_x.append(x); all_y.append(y)
        all_z.append(z); all_v.append(v)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z); all_v = np.concatenate(all_v)
    def padded(arr, frac=0.12, margin=0.5):
        p = (arr.max() - arr.min()) * frac + margin
        return arr.min() - p, arr.max() + p
    return (*padded(all_x), *padded(all_y), *padded(all_z),
            all_v.min(), all_v.max())


def make_frame(ax, traj, step, obj_val, waypoints, d, limits, azim, ghost_trajs=None):
    """Render one 3D frame; returns the current Line3DCollection for colorbar sync."""
    ax.clear()
    x_min, x_max, y_min, y_max, z_min, z_max, v_min, v_max = limits
    norm = plt.Normalize(v_min, v_max)

    # ── Ghost trails (previous iterations, fading out) ────────────────────
    if ghost_trajs:
        n_ghosts = len(ghost_trajs)
        alphas = np.linspace(GHOST_ALPHA_MIN, GHOST_ALPHA_MAX, n_ghosts)
        for ghost_traj, alpha in zip(ghost_trajs, alphas):
            gx, gy, gz, gv = trajectory_xyz_speed(ghost_traj, d)
            gpts = np.array([gx, gy, gz]).T.reshape(-1, 1, 3)
            gsegs = np.concatenate([gpts[:-1], gpts[1:]], axis=1)
            glc = Line3DCollection(gsegs, cmap="plasma", linewidth=1.8,
                                   norm=norm, alpha=alpha)
            glc.set_array(gv)
            ax.add_collection(glc)

    # ── Current trajectory (full opacity) ────────────────────────────────
    x, y, z, v = trajectory_xyz_speed(traj, d)
    points   = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap="plasma", linewidth=2.5,
                          norm=norm)
    lc.set_array(v)
    ax.add_collection(lc)

    # Waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               color="white", edgecolors="black", s=40, linewidths=1.0,
               depthshade=False, zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=CAMERA_ELEV, azim=azim)
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
    ax.set_xlabel("x [m]", fontsize=9, labelpad=4)
    ax.set_ylabel("y [m]", fontsize=9, labelpad=4)
    ax.set_zlabel("z [m]", fontsize=9, labelpad=4)
    ax.set_title(f"Step {step}   |   Obj = {obj_val:.3f}", fontsize=11, pad=8)
    ax.grid(True, alpha=0.2)
    return lc


def main():
    print("Running trajectory optimization …")
    solver = NoTimestampSolver(d=D, r=R, q=Q, dimensions=3)
    n_segs = len(WAYPOINTS) - 1
    x0 = np.full(n_segs, T_TOTAL / n_segs)
    obj, inter_results, inter_trajs, inter_times = solver.solve_backtracking_line_search(
        WAYPOINTS, T_TOTAL, intermediate_traj=True, x0=x0
    )

    if not inter_trajs:
        print("No intermediate trajectories recorded (converged in 0 steps).")
        return

    n_conv = len(inter_trajs)
    print(f"Optimization done: {n_conv} convergence steps, final obj = {obj:.4f}")

    # Map each saved trajectory to its objective value.
    # inter_results[i] is the objective at the START of outer iteration i;
    # inter_trajs stores the accepted trajectory at the end of that same
    # iteration, so pair them 1-to-1 (len may differ if last step is skipped).
    n_conv = len(inter_trajs)
    obj_vals = list(inter_results[:n_conv])
    while len(obj_vals) < n_conv:
        obj_vals.append(obj)

    # ── Frame layout ─────────────────────────────────────────────────────────
    # Camera spins exactly 360° total. Convergence loops continuously to fill
    # however many frames that requires. No separate spin-only phase.
    n_frames = int(360.0 / CAMERA_DEG_STEP)

    limits = compute_global_limits(inter_trajs, D)

    # Pre-cache xyz/speed for all trajectories (avoid recomputing every frame)
    traj_data = [trajectory_xyz_speed(t, D) for t in inter_trajs]

    # ── Build animation ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 7))
    fig.subplots_adjust(bottom=0.10)
    ax  = fig.add_subplot(111, projection="3d")
    cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])

    lc0  = make_frame(ax, inter_trajs[0], 1, obj_vals[0], WAYPOINTS, D, limits, 0)
    cbar = fig.colorbar(lc0, cax=cax, orientation="horizontal")
    cbar.set_label("Speed [m/s]", fontsize=10)

    def update(frame_idx):
        azim      = frame_idx * CAMERA_DEG_STEP          # 0 → 360°
        conv_step = (frame_idx // FRAMES_PER_STEP) % n_conv  # loops convergence
        ghost_start = max(0, conv_step - GHOST_STEPS)
        ghosts = inter_trajs[ghost_start:conv_step]  # oldest → newest
        lc = make_frame(
            ax, inter_trajs[conv_step],
            conv_step + 1, obj_vals[conv_step],
            WAYPOINTS, D, limits, azim,
            ghost_trajs=ghosts,
        )
        cbar.update_normal(lc)
        return (lc,)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // FPS, blit=False,
    )

    print(f"Saving '{OUTPUT_FILE}' ({n_frames} frames @ {FPS} fps) …")
    anim.save(OUTPUT_FILE, writer="pillow", fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f"Done → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
