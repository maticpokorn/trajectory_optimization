"""
Sweep over band widths across many random examples and plot how often
each band width results in a non-positive-definite Hessian approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
from source.solver import TimestampSolver

d, r, q, dimensions = 9, 4, 4, 3
N_WAYPOINTS = 20
N_EXAMPLES = 20
N_SEGMENTS = N_WAYPOINTS - 1


def run_example(seed):
    """Return array of min eigenvalues for each band width, or None if QP fails."""
    np.random.seed(seed)
    waypoints = np.random.rand(N_WAYPOINTS, 3)
    travel_time = float(N_WAYPOINTS)

    solver = TimestampSolver(d=d, r=r, q=q, dimensions=dimensions)
    dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    t = travel_time * dists / dists.sum()

    obj = solver.solve_sparse(waypoints, t)
    if obj == np.inf:
        return None

    min_eigs = []
    for bands in range(1, N_SEGMENTS + 1):
        _, H = solver.grad_hess2(waypoints, t, bands=bands)
        min_eigs.append(np.linalg.eigvalsh(H)[0])
    return min_eigs


# --- Collect results ---
# non_pd_counts[b] = number of examples where bands=b+1 gave non-PD result
non_pd_counts = np.zeros(N_SEGMENTS, dtype=int)
valid = 0

for seed in range(N_EXAMPLES):
    print(f"Running seed {seed+1}/{N_EXAMPLES}...", flush=True)
    result = run_example(seed)
    if result is None:
        print(f"  [SKIP] QP failed")
        continue
    valid += 1
    for b, min_eig in enumerate(result):
        if min_eig <= 1e-10:  # singular or not PD
            non_pd_counts[b] += 1

print(f"\nCompleted {valid}/{N_EXAMPLES} examples.\n")

# --- Print summary table ---
bands_range = np.arange(1, N_SEGMENTS + 1)
print(f"{'bands':>6}  {'non-PD count':>12}  {'fraction':>10}")
print("-" * 35)
for b, count in zip(bands_range, non_pd_counts):
    print(f"{b:>6}  {count:>12}  {count/valid:>10.2%}")

# --- Plot ---
fractions = non_pd_counts / valid

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(bands_range, fractions * 100, color="steelblue", edgecolor="black", linewidth=0.5)

# Highlight bands that are always PD
for bar, frac in zip(bars, fractions):
    if frac == 0:
        bar.set_color("green")
    elif frac == 1:
        bar.set_color("red")

ax.set_xlabel("Band width", fontsize=13)
ax.set_ylabel("Non-PD rate (%)", fontsize=13)
ax.set_title(
    f"Non-positive-definite Hessian rate by band width\n"
    f"({valid} random examples, {N_WAYPOINTS} waypoints, d={d}, r={r}, q={q})",
    fontsize=13,
)
ax.set_xticks(bands_range[::2])
ax.set_xlim(0.5, N_SEGMENTS + 0.5)
ax.set_ylim(0, 105)
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("img/band_width_non_pd_rate.pdf", bbox_inches="tight")
plt.show()

