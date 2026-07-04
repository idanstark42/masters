import numpy as np
import matplotlib.pyplot as plt

def distribution(mu, size=None):
  return np.random.pareto(mu, size=size) + 1.0

def main(n, mu_range, n_trials, qs_range):
  mu_values = np.arange(
    mu_range[0],
    mu_range[1] + mu_range[2] / 2,
    mu_range[2]
  )
  qs = np.linspace(*qs_range)

  plt.figure(figsize=(8, 5))
  for mu in mu_values:

    S = np.empty(n_trials)
    R = np.empty(n_trials)

    for i in range(n_trials):
      x = distribution(mu, size=n)
      s = np.sum(x)
      m = np.max(x)

      S[i] = s
      R[i] = m / s

    plt.plot(qs, [np.mean(R[S > np.quantile(S, q)]) for q in qs], marker="o", label=fr"$\mu={mu:.1f}$")

  plt.xlabel("Quantile threshold of $S_n$")
  plt.ylabel(r"$\mathbb{E}[M_n / S_n \mid S_n \text{ large}]$")
  plt.ylim(0, 1.05)
  plt.grid(alpha=0.3)
  plt.legend()
  plt.title(f"One-big-jump principle")
  plt.savefig('figures/part2.png')
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main(n=30, mu_range=(0.5, 1.5, 0.5), n_trials=100000, qs_range=(0.9, 0.999, 20))