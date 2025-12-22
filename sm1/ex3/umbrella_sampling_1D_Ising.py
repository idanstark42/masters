import math
from tqdm import tqdm
import logging

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
  
lcg = LinearCongruentialGenerator(seed=42)

def randint(a, b):
  return a + lcg.next() % (b - a + 1)

def random():
  return lcg.next() / lcg.modulus

# section: physics

def hamiltonian(state, J, m0, bias = 0, boundary_condition='open'):
  spins = state.spins
  m = state.m()
  if boundary_condition == 'periodic':
    return -J * sum(spins[i] * spins[(i + 1) % len(spins)] for i in range(len(spins))) - (1/2) * bias * (m - m0) ** 2
  else:
    return -J * sum(spins[i] * spins[i + 1] for i in range(len(spins) - 1))

def random_initial_state(N):
  num = randint(0, 2 ** N)
  return State([1 if (num >> i) & 1 else -1 for i in range(N)])

# section: simulations

class State:
  def __init__(self, spins):
    self.spins = spins
    self.size = len(spins)

  def m (self):
    return sum(self.spins) / self.size

class Dynamics:
  pass

  def next(self, state):
    return state

class SingleSpinMetropolis(Dynamics):
  def __init__(self, T, hamiltonian_func):
    self.T = T
    self.hamiltonian_func = hamiltonian_func

  def next(self, state):
    spins = state.spins[:] 
    flipped_state = State(spins)
    i = randint(0, state.size - 1)
    flipped_state.spins[i] *= -1  # Flip the spin at position i
    dE = self.hamiltonian_func(flipped_state) - self.hamiltonian_func(state)
    if dE < 0 or random() < math.exp(-dE / self.T):
      return flipped_state

def monte_carlo_sweep(state, dynamics):
  for _ in range(state.size):
    state = dynamics.next(state)
  return state

def monte_carlo (initial_state_generator, dynamics, value, sweeps_after_convergence, convergence):
  states = initial_state_generator()
  values_history = [[value(state) for state in states]]

  # run until convergence
  while True:
    states = [monte_carlo_sweep(state, dynamics) for state in states]
    values_history.append([value(state) for state in states])
    if convergence.converge(values_history):
      break

  # run some more sweeps to get final average
  for _ in range(sweeps_after_convergence):
    states = [monte_carlo_sweep(state, dynamics) for state in states]
    values_history.append([value(state) for state in states])
  return sum(value(state) for state in states) / len(states), values_history

# section: convergence criteria

class Convergence:
  def converge(self, values_history):
    pass

class VarianceConvergence(Convergence):
  def __init__(self, threshold, repetitions):
    self.threshold = threshold
    self.repetitions = repetitions

  def converge(self, values_history):
    if len(values_history) < self.repetitions:
      return False
    recent_averages = [sum(values_history[-i]) / len(values_history[-i]) for i in range(1, self.repetitions + 1)]
    mean = sum(recent_averages) / self.repetitions
    variance = sum((x - mean) ** 2 for x in recent_averages) / self.repetitions
    return variance < self.threshold

class SlopeConvergence(Convergence):
  def __init__(self, threshold, repetitions):
    self.threshold = threshold
    self.repetitions = repetitions

  def converge(self, values_history):
    if len(values_history) < self.repetitions:
      return False
    recent_averages = [sum(values_history[-i]) / len(values_history[-i]) for i in range(1, self.repetitions + 1)]
    n = self.repetitions
    sum_x = sum(range(n))
    sum_y = sum(recent_averages)
    sum_xx = sum(x * x for x in range(n))
    sum_xy = sum(x * recent_averages[x] for x in range(n))
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return abs(slope) < self.threshold

class MultipleConvergence(Convergence):
  def __init__(self, convergences):
    self.convergences = convergences

  def converge(self, values_history):
    return all(c.converge(values_history) for c in self.convergences)

# section: exercises

def question1(N, J, T):
  n_sweeps = 1000
  n_samples = 10
  dynamics = SingleSpinMetropolis(T, lambda s: hamiltonian(s, J, 0, boundary_condition='periodic'))
  convergence = MultipleConvergence([
    VarianceConvergence(threshold=1e-4, repetitions=5),
    SlopeConvergence(threshold=1e-4, repetitions=5)
  ])
  result = monte_carlo(lambda: [random_initial_state(N) for _ in range(n_samples)], dynamics, lambda s: s.m(), n_sweeps, convergence)

def main(q):
  N = 128
  J = 1.0
  T = 2.5
  if q == 1:
    question1(N, J, T)

if __name__ == "__main__":
  import sys
  ex = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  main(ex)