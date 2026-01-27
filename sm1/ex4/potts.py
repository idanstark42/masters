import math
import numpy as np
from scipy.stats import truncnorm, norm
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from tqdm import trange
from itertools import product
from scipy.signal import find_peaks

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# section: statistics

def fit_from_histogram(data, bins=30, peak_prominence=0.05):
  # Compute histogram
  hist, bin_edges = np.histogram(data, bins=bins)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

  # Find peaks
  peaks_idx, _ = find_peaks(hist, prominence=peak_prominence)
  
  if len(peaks_idx) < 2:
    # Unimodal: fit one Gaussian
    mu, sigma = norm.fit(data)
    return mu, sigma
  
  # Take two highest peaks
  top_two = np.argsort(hist[peaks_idx])[-2:]
  left_peak_idx = peaks_idx[top_two[0]]
  right_peak_idx = peaks_idx[top_two[1]]
  # order them
  if left_peak_idx > right_peak_idx:
    left_peak_idx, right_peak_idx = right_peak_idx, left_peak_idx

  # Find minimum between peaks
  min_idx = np.argmin(hist[left_peak_idx:right_peak_idx + 1]) + left_peak_idx

  split_value = bin_centers[min_idx]

  # Split data
  left_data = [datum for datum in data if datum <= split_value]
  right_data = [datum for datum in data if datum > split_value]

  # return the mean and std of the higher peak
  if hist[left_peak_idx] >= hist[right_peak_idx]:
    return norm.fit(left_data)
  else:
    return norm.fit(right_data)

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

# section: physics

def delta(a, b):
  return 1 if a == b else 0

def hamiltonian(state, J, boundary_condition='open'):
  spins = state.spins
  if boundary_condition == 'periodic':
    return -J * sum(delta(spins[i], spins[(i + 1) % len(spins)]) for i in range(len(spins)))
  elif boundary_condition == 'open':
    return -J * sum(delta(spins[i], spins[i + 1]) for i in range(len(spins) - 1))
  else:
    raise ValueError("Unknown boundary condition: {}".format(boundary_condition))

def random_initial_state(N, q):
  return State([random.integer(1, q) for _ in range(N)], q)

def physical_initial_state(N, q):
  random_magnetizations = [random.fraction() for _ in range(q)]

  # randomly bias one of the magnetizations to be larger
  bias = (random.fraction() * 4) ** 2
  random_magnetizations[random.integer(0, q - 1)] *= np.exp(bias)

  total_magnetization = sum(random_magnetizations)
  m_ks = [m / total_magnetization for m in random_magnetizations]
  spins = [k for k in range(1, q + 1) for _ in range(int(N * m_ks[k - 1]))]
  if len(spins) < N:
    spins += [random.integer(1, q) for _ in range(N - len(spins))]
  random.shuffle(spins)
  return State(spins, q)

# section: simulations

class State:
  def __init__(self, spins, q):
    self.spins = spins
    self.size = len(spins)
    self.q = q

  def copy(self):
    return State(self.spins[:], self.q)

  def m_k(self, k):
    return sum(delta(s, k) for s in self.spins) / self.size

  def order_parameter(self):
    m_ks = [self.m_k(k) for k in range(1, self.q + 1)]
    return np.std(m_ks)
  
  def __repr__(self):
    return ''.join([str(s) for s in self.spins])

class Dynamics:
  pass

  def next(self, state):
    return state

class SingleSpinMetropolis(Dynamics):
  def __init__(self, q, T, hamiltonian):
    self.q = q
    self.T = T
    self.hamiltonian = hamiltonian

  def different_spin(self, current_spin):
    diff = random.integer(1, self.q - 1)
    return ((current_spin - 1 + diff) % self.q) + 1

  def next(self, state):
    flipped_state = state.copy()
    i = random.integer(0, flipped_state.size - 1)
    flipped_state.spins[i] = self.different_spin(flipped_state.spins[i])
    dE = self.hamiltonian(flipped_state) - self.hamiltonian(state)
    return flipped_state if dE < 0 or random.fraction() < math.exp(-dE / self.T) else state

class MonteCarlo:
  def __init__(self, initial_state_generator, dynamics, value, sweeps_after_convergence, convergence_criterion, q, blind=False):
    self.initial_state_generator = initial_state_generator
    self.dynamics = dynamics
    self.value = value
    self.sweeps_after_convergence = sweeps_after_convergence
    self.convergence_criterion = convergence_criterion
    self.blind = blind
    self.q = q

  def init(self):
    logger.info("Initializing Monte Carlo simulation")
    self.states = self.initial_state_generator()
    self.history = [[self.value(state) for state in self.states]]

  def sweep(self, state):
    index = self.states.index(state)
    logger.debug("sweep {} {}".format(index, state))
    for _ in range(state.size):
      state = self.dynamics.next(state)
    logger.debug("swept {} {}".format(index, state))
    return state
  
  def step(self):
    self.states = [self.sweep(state) for state in self.states]
    self.history.append([self.value(state) for state in self.states])
    if not self.blind:
      self.plot_history()

  def run(self):
    logger.info("Starting Monte Carlo simulation")
    self.init()
    # run until convergence
    while not self.convergence_criterion.converged(self.history):
      self.step()

    logger.info("Convergence reached after {} sweeps".format(len(self.history)))

    # run some more sweeps to get final average
    for _ in trange(self.sweeps_after_convergence, desc="Post-convergence sweeps", unit="sweep", ncols=80):
      self.step()

    logger.info("Finished Monte Carlo simulation")
    # take all sweeps since convergence and average the value
    sweeps_since_convergence = self.history[-(self.sweeps_after_convergence + 1):-1]
    values_since_convergence = [v for sweep in sweeps_since_convergence for v in sweep]

    average = np.mean(values_since_convergence)
    std_dev = np.std(values_since_convergence)

    self.plot_history(save=True)
    # take N bins exactly around the zero
    h = 2 / (self.states[0].size)
    bin_count = int(0.5 / h)
    bins = [i * h for i in range(bin_count + 1)]
    self.plot_histogram(n_sweeps=self.sweeps_after_convergence, bins=bins)

    return average, std_dev, values_since_convergence, self.history

  def plot_history(self, save=False):
    plt.clf()
    averages = [np.mean(h) for h in self.history]
    stds = [np.std(h) for h in self.history]
    tops = [a + s for a, s in zip(averages, stds)]
    bottoms = [a - s for a, s in zip(averages, stds)]
    plt.fill_between(range(len(averages)), bottoms, tops, color='lightblue', alpha=0.5, label='Std Dev')
    plt.plot(averages, label='Average')
    plt.xlabel('Sweep')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.title('Monte Carlo Simulation Progress')
    plt.pause(0.1)
    if save:
      plt.savefig('graphs/monte_carlo_history_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

  # take the last n sweeps and plot histogram
  def plot_histogram(self, n_sweeps=10, **kwargs):
    plt.clf()
    sweeps = self.history[-n_sweeps:]
    values = [v for sweep in sweeps for v in sweep]
    plt.hist(values, **kwargs)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Histogram of Values from Monte Carlo Simulation')
    plt.pause(0.1)
    # save figure by datetime
    plt.savefig('graphs/monte_carlo_histogram_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
# section: convergence criteria

class Convergence:
  def converged(self, history):
    return False

# requires at least `repetitions` recent averages to have variance below `threshold`
class StandardDeviationConvergence(Convergence):
  def __init__(self, threshold, repetitions):
    self.threshold = threshold
    self.repetitions = repetitions

  def converged(self, history):
    if len(history) < self.repetitions:
      return False
    recent_averages = [np.mean(history[-i]) for i in range(1, self.repetitions + 1)]
    return np.std(recent_averages) < self.threshold

# requires at least `repetitions` recent averages to have slope below `threshold`, i.e. be stable
class SlopeConvergence(Convergence):
  def __init__(self, threshold, repetitions):
    self.threshold = threshold
    self.repetitions = repetitions

  def converged(self, history):
    if len(history) < self.repetitions:
      return False
    recent_averages = [np.mean(history[-i]) for i in range(1, self.repetitions + 1)]
    # compute slope using linear regression
    slope = np.polyfit(range(self.repetitions), recent_averages, 1)[0]
    return abs(slope) < self.threshold

class MultipleConvergence(Convergence):
  def __init__(self, convergences):
    self.convergences = convergences

  def converged(self, history):
    return all(c.converged(history) for c in self.convergences)

# section: main
# run a simulation of the Potts model with q states, coupling J, temperature T, on a chain of size N
def simulate(q, T):
  N = 180
  J = 1.0

  num_states = 5  # number of parallel states to simulate
  sweeps_after_convergence = 200  # number of sweeps to run after convergence
  convergence_criterion = MultipleConvergence([
    StandardDeviationConvergence(threshold=0.1, repetitions=50),
    SlopeConvergence(threshold=0.001, repetitions=50)
  ])

  def initial_state_generator():
    states = []
    for _ in range(num_states):
      states.append(physical_initial_state(N, q))
    return states
      
  dynamics = SingleSpinMetropolis(T=T, q=q, hamiltonian=lambda state: hamiltonian(state, J, boundary_condition='periodic'))

  monte_carlo = MonteCarlo(
    initial_state_generator=initial_state_generator,
    dynamics=dynamics,
    value=lambda state: state.order_parameter(),
    sweeps_after_convergence=sweeps_after_convergence,
    convergence_criterion=convergence_criterion,
    q=q,
    blind=False
  )

  average, stdev, values, history = monte_carlo.run()
  h = 2 / N
  bin_count = int(0.5 / h)
  bins = [i * h for i in range(bin_count + 1)]
  mu, std = fit_from_histogram(values, bins=bins, peak_prominence=0.05)
  return mu, std

def main(q):
  # run for T starting from 0 until you each the disordered phase for at least three points, jumps of 0.2
  T_MIN = 0.02
  T = T_MIN
  dT = 0.02
  T_MAX = 0.5
  values, stds = [], []
  while True:
    avg, std = simulate(q, T)
    values.append(max(avg, 0.0))  # avoid negative values due to fitting errors
    stds.append(std)
    logger.info("T = {:.2f}, Order Parameter = {:.6f} ± {:.6f}".format(T, avg, std))
    if T > T_MAX:
      break
    T += dT
  # plot the order parameter as a function of T
  logger.info("Values: {}, stds: {}".format(values, stds))
  plt.clf()
  plt.errorbar([T_MIN + dT * i for i in range(len(values))], values, yerr=stds, fmt='o-')
  plt.xlabel('Temperature T')
  plt.ylabel('Order Parameter')
  plt.title('Order Parameter vs Temperature for q = {}'.format(q))
  plt.savefig('graphs/order_parameter_q_{}_{}.png'.format(q, datetime.now().strftime('%Y%m%d_%H%M%S')))
  plt.show()

if __name__ == "__main__":
  import sys
  q = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  main(q)