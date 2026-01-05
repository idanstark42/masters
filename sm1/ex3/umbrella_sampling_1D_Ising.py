import math
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from tqdm import trange

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

# section: physics

def hamiltonian(state, J, m0, k = 0, boundary_condition='open'):
  spins = state.spins
  m = state.m()
  if boundary_condition == 'periodic':
    return -J * sum(spins[i] * spins[(i + 1) % len(spins)] for i in range(len(spins))) + (1/2) * k * (m - m0) ** 2
  elif boundary_condition == 'open':
    return -J * sum(spins[i] * spins[i + 1] for i in range(len(spins) - 1)) + (1/2) * k * (m - m0) ** 2
  else:
    raise ValueError("Unknown boundary condition: {}".format(boundary_condition))

def random_initial_state(N):
  return State([1 if random.fraction() < 0.5 else -1 for _ in range(N)])

# generates a random initial state based on a magnetization
def physical_initial_state(N, m):
  n_up = int((m + 1) / 2 * N)
  n_down = N - n_up
  spins = [1] * n_up + [-1] * n_down
  random.shuffle(spins)
  return State(spins)

# section: simulations

class State:
  def __init__(self, spins):
    self.spins = spins
    self.size = len(spins)

  def copy(self):
    return State(self.spins[:])

  def m(self):
    return np.sum(self.spins) / self.size
  
  def __repr__(self):
    return ''.join(['↑' if s == 1 else '↓' for s in self.spins])

class Dynamics:
  pass

  def next(self, state):
    return state

class SingleSpinMetropolis(Dynamics):
  def __init__(self, T, hamiltonian):
    self.T = T
    self.hamiltonian = hamiltonian

  def next(self, state):
    flipped_state = state.copy()
    i = random.integer(0, flipped_state.size - 1)
    flipped_state.spins[i] *= -1  # Flip the spin at position i
    dE = self.hamiltonian(flipped_state) - self.hamiltonian(state)
    return flipped_state if dE < 0 or random.fraction() < math.exp(-dE / self.T) else state

class MonteCarlo:
  def __init__(self, initial_state_generator, dynamics, value, sweeps_after_convergence, convergence_criterion, blind=False):
    self.initial_state_generator = initial_state_generator
    self.dynamics = dynamics
    self.value = value
    self.sweeps_after_convergence = sweeps_after_convergence
    self.convergence_criterion = convergence_criterion
    self.blind = blind

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

    logger.info("Average: {:.6f}".format(average))
    logger.info("Std Dev: {:.6f}".format(std_dev))

    self.plot_history(save=True)
    # take N bins exactly around the zero
    h = 2 / self.states[0].size
    bin_count = int(2 / h)
    bins = [-1 + i * h for i in range(bin_count + 1)]
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
    plt.ylim(-1, 1)
    plt.title('Monte Carlo Simulation Progress')
    plt.pause(0.1)
    if save:
      plt.savefig('monte_carlo_history_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

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
    plt.savefig('monte_carlo_histogram_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
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

# section: exercises

def question1(N, J, T):
  logger.info("Running Question 1: N={}, J={}, T={}".format(N, J, T))
  sweeps_after_convergence = 1000
  n_samples = 20
  dynamics = SingleSpinMetropolis(T, lambda s: hamiltonian(s, J, 0, boundary_condition='periodic'))
  convergence = MultipleConvergence([
    StandardDeviationConvergence(threshold=1, repetitions=10),
    SlopeConvergence(threshold=0.001, repetitions=10)
  ])
  # generate initial states with random magnetizations instead of random spins
  initial_state_generator = lambda: [physical_initial_state(N, random.fraction() * 2 - 1) for _ in range(n_samples)]
  monte_carlo = MonteCarlo(initial_state_generator, dynamics, lambda s: s.m(), sweeps_after_convergence, convergence, blind=True)
  average, std_dev, values_since_convergence, history = monte_carlo.run()

  # print the number and percent of values outside the [-0.5, 0.5] range
  n_outside = sum(1 for v in values_since_convergence if abs(v) > 0.5)
  logger.info("{} values outside [-0.5, 0.5] out of {} range ({:.2f}%)".format(n_outside, len(values_since_convergence), n_outside / len(values_since_convergence) * 100))

def question2(N, J, T):
  logger.info("Running Question 2: N={}, J={}, T={}".format(N, J, T))
  sweeps_after_convergence = 1000
  n_samples = 20
  bias = 2000
  m_0 = 0.6
  dynamics = SingleSpinMetropolis(T, lambda s: hamiltonian(s, J, m_0, k=bias, boundary_condition='periodic'))
  convergence = MultipleConvergence([
    StandardDeviationConvergence(threshold=1, repetitions=10),
    SlopeConvergence(threshold=0.001, repetitions=10)
  ])
  # generate initial states with random magnetizations instead of random spins
  initial_state_generator = lambda: [physical_initial_state(N, random.fraction() * 2 - 1) for _ in range(n_samples)]
  monte_carlo = MonteCarlo(initial_state_generator, dynamics, lambda s: s.m(), sweeps_after_convergence, convergence, blind=True)
  average, std_dev, values_since_convergence, history = monte_carlo.run()

  # print the number and percent of values outside the [-0.5, 0.5] range
  n_outside = sum(1 for v in values_since_convergence if abs(v) > 0.5)
  logger.info("{} values outside [-0.5, 0.5] out of {} range ({:.2f}%)".format(n_outside, len(values_since_convergence), n_outside / len(values_since_convergence) * 100))

# same as question 2 but take the sample and multiply each p(m) by exp(W(m) / T ) to unbias the distribution, then plot the histogram
def question3(N, J, T):
  logger.info("Running Question 3")
  sweeps_after_convergence = 1000
  n_samples = 20
  bias = 2000
  m_0 = 0.6
  dynamics = SingleSpinMetropolis(T, lambda s: hamiltonian(s, J, m_0, k=bias, boundary_condition='periodic'))
  convergence = MultipleConvergence([
    StandardDeviationConvergence(threshold=1, repetitions=10),
    SlopeConvergence(threshold=0.001, repetitions=10)
  ])
  # generate initial states with random magnetizations instead of random spins
  initial_state_generator = lambda: [physical_initial_state(N, random.fraction() * 2 - 1) for _ in range(n_samples)]
  monte_carlo = MonteCarlo(initial_state_generator, dynamics, lambda s: s.m(), sweeps_after_convergence, convergence, blind=True)
  average, std_dev, values_since_convergence, history = monte_carlo.run()
  # unbias the histogram
  h = 2 / N
  bin_count = int(2 / h)
  bins = [-1 + i * h for i in range(bin_count + 1)]
  histogram, bin_edges = np.histogram(values_since_convergence, bins=bins)
  bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
  weights = np.exp((1/2) * bias * (bin_centers - m_0) ** 2 / T)
  # remove infinities and nans from weights
  weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
  unbiased_histogram = histogram * weights
  plt.clf()
  plt.bar(bin_centers, unbiased_histogram, width=h, align='center')
  plt.xlabel('Magnetization m')
  plt.ylabel('Unbiased Count')
  plt.title('Unbiased Histogram of Magnetization from Umbrella Sampling')
  plt.pause(0.1)
  plt.savefig('unbiased_magnetization_histogram_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
  # calcualte the direct sampling histogram for comparison
  direct_monte_carlo = MonteCarlo(
    lambda: [physical_initial_state(N, random.fraction() * 2 - 1) for _ in range(n_samples)],
    SingleSpinMetropolis(T, lambda s: hamiltonian(s, J, 0, boundary_condition='periodic')),
    lambda s: s.m(),
    sweeps_after_convergence,
    convergence,
    blind=True
  )
  direct_average, direct_std_dev, direct_values_since_convergence, direct_history = direct_monte_carlo.run()
  direct_histogram, _ = np.histogram(direct_values_since_convergence, bins=bins)
  # normalize histograms
  direct_histogram = direct_histogram / np.sum(direct_histogram)
  unbiased_histogram = unbiased_histogram / np.sum(unbiased_histogram)
  # fit the unbiased histogram to a truncated normal distribution for better comparison
  # step 1 - find the fit parameters
  mu, std = truncnorm.fit(unbiased_histogram, floc=-1, fscale=2)[2:]
  # step 2 - find the fit maximum - the value at the mu
  unbiased_max = pow(math.e, -0.5 * ((mu - mu) / std) ** 2) / (std * math.sqrt(2 * math.pi))
  # step 3 - scale the unbiased histogram to match the direct histogram at the peak
  scale_factor = np.max(direct_histogram) / unbiased_max
  unbiased_histogram = unbiased_histogram * scale_factor
  # plot both histograms on the same graph
  plt.clf()
  plt.bar(bin_centers, direct_histogram, width=h, align='center', label='Direct Sampling')
  plt.bar(bin_centers, unbiased_histogram, width=h, align='center', label='Umbrella Sampling Unbiased')
  plt.xlabel('Magnetization m')
  plt.ylabel('Normalized Count')
  plt.title('Comparison of Direct Sampling and Umbrella Sampling (Unbiased)')
  plt.legend()
  plt.pause(0.1)
  plt.savefig('direct_vs_umbrella_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))


# section: main

# show distribution of random numbers
def test_random():
  samples = 100000
  values = [random.fraction() for _ in range(samples)]
  plt.clf()
  plt.hist(values, bins=50, density=True)
  plt.xlabel('Value')
  plt.ylabel('Probability Density')
  plt.title('Histogram of Random Numbers from LCG')
  plt.pause(0.1)
  plt.show()

def main(q):
  N = 128
  J = 1.0
  T = 2.5
  if q == 1:
    question1(N, J, T)
  elif q == 2:
    question2(N, J, T)
  elif q == 3:
    question3(N, J, T)
  elif q == 0:
    test_random()

if __name__ == "__main__":
  import sys
  q = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  main(q)