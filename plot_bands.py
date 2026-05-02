import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

df = pd.read_csv("bands_convergence_n30.csv")
bands = [1, 2, 3, 4, 5, 7, 10]

# Create a colormap with distinct colors
cmap = cm.tab10

# Different markers for each band
markers = ["o", "s", "^", "v", "D", "*", "p"]

for idx, i in enumerate(bands):
    color = cmap(idx)
    marker = markers[idx % len(markers)]
    plt.plot(
        df["iteration"],
        df[f"q{i}_mean"],
        label=f"Band {i}",
        color=color,
        alpha=0.7,
        marker=marker,
        markersize=4,
        markevery=5,
    )
plt.plot(
    df["iteration"],
    df["grad_mean"],
    label="grad",
    linestyle="--",
    color="red",
    alpha=0.7,
)
plt.legend()
plt.show()
