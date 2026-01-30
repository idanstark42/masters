import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import deque

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# section: random number generation

class LinearCongruentialGenerator:
  def __init__(self, seed=1):
    self.modulus = 2**31
    self.multiplier = 1103515245
    self.increment = 12345
    self.state = seed

  def next(self):
    self.state = (self.multiplier * self.state + self.increment) % self.modulus
    return self.state

  def integer(self, a, b):
    if a > b:
      raise ValueError("Invalid range: a must be less than or equal to b")
    if a == b:
      return a
    if b > a + self.modulus - 1:
      raise ValueError("Range too large: b - a must be less than modulus")
    if a == 0 and b == self.modulus - 1:
      return self.next()
    return a + self.next() % (b - a + 1)
  
  def fraction(self):
    return self.next() / self.modulus
  
  def shuffle(self, array):
    for i in range(len(array) - 1, 0, -1):
      j = self.integer(0, i)
      array[i], array[j] = array[j], array[i]
    return array

random = LinearCongruentialGenerator(seed=42)

# section: lattice model

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def neighboor_of(self, other):
    return (self.x == other.x and abs(self.y - other.y) == 1) or (self.y == other.y and abs(self.x - other.x) == 1)

  def __repr__(self):
    return '({},{})'.format(self.x, self.y)

  def __eq__(self, other):
    return isinstance(other, Point) and self.x == other.x and self.y == other.y

class Lattice:
  def __init__(self, size=2, p=0.5):
    self.size = size
    self.p = p
    self.generate()
    self._graph = None

  def generate(self):
    self.matrix = np.zeros((self.size, self.size), dtype=np.uint8)
    for x in range(self.size):
      for y in range(self.size):
        if(random.fraction() < self.p):
          self.matrix[y, x] = 1

  def __repr__(self):
    return '\n'.join([''.join([' + ' if s == 1 else '   ' for s in row]) for row in self.matrix])

  def occupation_density(self):
    return self.matrix.sum() / self.matrix.size

  def is_percolating(self):
    size = self.size
    visited = np.zeros((size, size), dtype=np.uint8)
    queue = deque()

    # Start BFS from all occupied cells in the top row
    top = np.where(self.matrix[0] == 1)[0]
    for x in top:
      queue.append((0, x))
      visited[0, x] = True

    # BFS
    while queue:
      y, x = queue.popleft()
      if y == size - 1:  # reached bottom row
        return True
      for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:  # 4 neighbors
        ny, nx = y + dy, x + dx
        if 0 <= ny < size and 0 <= nx < size:
          if self.matrix[ny, nx] == 1 and not visited[ny, nx]:
            visited[ny, nx] = True
            queue.append((ny, nx))
    return False

  def coarse_grain(self, factor):
    assert self.size % factor == 0, "size must be divisible by factor"

    new_size = self.size // factor
    coarse = Lattice.__new__(Lattice)   # bypass __init__
    coarse.size = new_size
    coarse.p = None
    coarse.matrix = np.zeros((new_size, new_size), dtype=np.uint8)

    for by in range(new_size):
      for bx in range(new_size):
        # extract block
        block_matrix = self.matrix[
          by*factor:(by+1)*factor,
          bx*factor:(bx+1)*factor
        ]

        # build block lattice
        block = Lattice.__new__(Lattice)
        block.size = factor
        block.p = None
        block.matrix = block_matrix

        # use existing percolation logic
        coarse.matrix[by, bx] = 1 if block.is_percolating() else 0

    return coarse

def percolation_probability(size = 2, p = 0.5, N = 100):
  return sum([1 if Lattice(size, p).is_percolating() else 0 for _ in trange(N, desc=f"testing size={size} p={p:.2f}", unit="cell", ncols=80)]) / N

def question1():
  size = 2048
  min_p = 0.0
  max_p = 1.0
  p_step = 0.05
  probabilities = []
  for p in np.arange(min_p, max_p, p_step):
    probability = percolation_probability(size, p)
    logger.info(f"p={p:.2f} => P={probability}")
    probabilities.append(probability)
  
  xs = np.linspace(min_p, max_p, int((max_p - min_p) / p_step))

  plt.plot(xs, probabilities)
  plt.xlabel('Site Occupation Probability')
  plt.ylabel('Precolation Probability')
  plt.title('Precolation as a function of Site Occupation for a 2048x2048 lattice')
  plt.show()

def plot_lattice(lattice, title=""):
  plt.figure(figsize=(4, 4))
  plt.imshow(lattice.matrix, cmap="gray", interpolation="nearest")
  plt.xticks([])
  plt.yticks([])
  plt.title(title)
  plt.tight_layout()
  plt.savefig(f"images/{title}.png", dpi=300)
  plt.close()

def question2():
  ps = { "p_c": 0.618, "0.55": 0.55, "0.65": 0.65 }
  for p_name, p in ps.items():
    logger.info(f"Starting run for p={p_name}")
    lattice = Lattice(size=2048, p=p)
    densities = [lattice.occupation_density()]
    plot_lattice(lattice, title=f"Original lattice with p={p_name}")    

    for i in trange(int(math.log2(lattice.size)) - 2, unit="decimation", ncols=80):
      lattice = lattice.coarse_grain(2)
      densities.append(lattice.occupation_density())
      plot_lattice(lattice, title=f"Lattice with p={p_name} after {i+1} decimations")
    
    logger.info("  Saving progress graph")
    decimations = range(len(densities))
    plt.plot(decimations, densities)
    plt.title(f"Lattice density as a function of deimations for p={p_name}")
    plt.xlabel("Decimations")
    plt.ylabel("Lattice density")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f"images/Lattice density as a function of deimations for p={p_name}.png", dpi=300)
    plt.close()

def run_average_for_p_c():
  p_c = 0.618
  repetitions = 5
  densities = [None for _ in range(repetitions)]
  for repetition in range(repetitions):
    logger.info(f"Starting run #{repetition+1}")
    lattice = Lattice(size=2048, p=p_c)
    densities[repetition] = [lattice.occupation_density()]

    for i in trange(int(math.log2(lattice.size)) - 2, unit="decimation", ncols=80):
      lattice = lattice.coarse_grain(2)
      densities[repetition].append(lattice.occupation_density())

  average_densities = [sum([densities[i][j] for i in range(len(densities))]) / len(densities) for j in range(len(densities[0]))]
  decimations = range(len(average_densities))
  plt.plot(decimations, average_densities)
  plt.title(f"Lattice density as a function of deimations for p=p_c")
  plt.xlabel("Decimations")
  plt.ylabel("Lattice density")
  plt.ylim(0,1)
  plt.tight_layout()
  plt.savefig(f"images/Average lattice density as a function of deimations for p=p_c.png", dpi=300)
  plt.close()

def main(q):
  if q == 1:
    question1()
  elif q == 2:
    question2()
  elif q == 3:
    run_average_for_p_c()
  
if __name__ == "__main__":
  import sys
  q = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  main(q)
