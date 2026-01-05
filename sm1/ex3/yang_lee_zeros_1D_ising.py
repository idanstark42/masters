import sys
import numpy as np
import matplotlib.pyplot as plt
from math import comb

def rho (k, N, sign, TAU):
  phi = np.pi * (2*k + 1) / N
  tan2 = np.tan(phi / 2) ** 2
  B = (4 * TAU ** 2 - 2 + 2 * tan2) / (1 + tan2)
  rho = (- B + sign * (B ** 2 - 4+ 0j) ** 0.5) / 2
  print(k, phi, tan2, B, rho)
  return rho

def plot_for_n (N, TAU):
  rhos_plus = np.array([rho(k, N, +1, TAU) for k in range(0, N) if 2*k + 1 != N + 1])
  rhos_minus = np.array([rho(k, N, -1, TAU) for k in range(0, N) if 2*k + 1 != N + 1])
  rhos = np.concatenate((rhos_plus, rhos_minus))
  plt.scatter(rhos.real, rhos.imag, s=10)

  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)

  # center around origin while including all points
  max_extent = np.max(np.abs(np.concatenate([rhos.real, rhos.imag])))
  margin = 1.05
  lim = margin * max_extent
  plt.xlim(-lim, lim)
  plt.ylim(-lim, lim)

  plt.xlabel(r"$\Re(\rho)$")
  plt.ylabel(r"$\Im(\rho)$")
  plt.title(rf"Zeros of Z for $N={N}$")
  plt.gca().set_aspect("equal", adjustable="box")
  plt.savefig(f"rho_scatter_N{N}_tau_{TAU}.png", dpi=300, bbox_inches="tight")
  plt.show()

if __name__ == "__main__":
  N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
  TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
  plot_for_n(N, TAU)
