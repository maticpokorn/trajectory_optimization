import numpy as np
import matplotlib.pyplot as plt
from source.solver import banded


def banded(M, k=2):
    n, m = M.shape
    i, j = np.ogrid[:n, :m]
    mask = np.abs(i - j) < k
    return M * mask


def zero_mask(k, n):
    i, j = np.ogrid[:n, :n]
    mask = np.abs(i - j) < k
    return mask


H = np.load("hessian.npy")
plt.imshow(H)
plt.axis('off')
plt.savefig("img/hessian_full.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
for i in range(5, 0, -1):
    H_banded = banded(H, i)
    plt.imshow(H_banded)
    plt.axis('off')
    plt.savefig(f"img/hessian_band_{i}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
