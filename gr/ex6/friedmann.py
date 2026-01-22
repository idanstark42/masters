from sympy import Function, simplify
from sympy import symbols
from sympy.diffgeom import metric_to_Riemann_components, metric_to_Ricci_components, TensorProduct, Manifold, Patch, CoordSystem, metric_to_Riemann_components, TensorProduct
from sympy.functions import sin
from sympy.matrices import diag

c = symbols('c')  # Speed of light
pi = symbols('pi')  # Pi constant

def flatten(lst):
  return [item for sublist in lst for item in sublist]

def friedmann_metric_components(a, k):
  t, r, theta, phi = symbols('t r theta phi')
  return diag(
      -1,
      a**2/(1 - k*r**2),
      a**2*r**2,
      a**2*r**2*sin(theta)**2
  )

def friedmann_metric (a, k):
  # Define the manifold and coordinate system
  M = Manifold('M', 4)
  P = Patch('P', M)
  t, r, theta, phi = symbols('t r theta phi')
  coords = CoordSystem('coords', P, [t, r, theta, phi])
  t, r, theta, phi = coords.coord_functions()
  oneforms = coords.base_oneforms()

  # Metric components
  components = friedmann_metric_components(a, k)

  # Construct the metric tensor
  return sum([sum([components[i, j] * TensorProduct(oneforms[i], oneforms[j]) for j in range(4)]) for i in range(4)])

def friedmann_riemann (a, k):
  g = friedmann_metric(a, k)
  Riemann = metric_to_Riemann_components(g)
  return Riemann

def friedmann_ricci (a, k):
  g = friedmann_metric(a, k)
  Ricci = metric_to_Ricci_components(g)
  return Ricci

def friedmann_scalar_curvature (a, k):
  Ricci = friedmann_ricci(a, k)
  g = friedmann_metric_components(a, k)
  return sum([g[i, j] * Ricci[i, j] for i in range(4) for j in range(4)])

def friendmann_einstein_tensor (a, k):
  Ricci = friedmann_ricci(a, k)
  R = friedmann_scalar_curvature(a, k)
  g = friedmann_metric_components(a, k)

  return [[Ricci[i, j] - 0.5 * R * g[i, j] for j in range(4)] for i in range(4)]

def friedmann_energy_momentum_tensor (a, k, rho, p, u):
  g = friedmann_metric_components(a, k)
  return [[(rho + p / c ** 2) * u[i] * u[j] - p * g[i, j] for j in range(4)] for i in range(4)]

def friedmann_equations (a, k, rho, p):
  G = friendmann_einstein_tensor(a, k)
  T = friedmann_energy_momentum_tensor(a, k, rho, p, [1, 0, 0, 0])

  return flatten([[G[i][j] - (8 * pi / c ** 4) * T[i][j] for j in range(4)] for i in range(4)])

if __name__ == "__main__":
  t, k, rho, p = symbols('t k rho p')
  a = Function('a')(t)

  equations = friedmann_equations(a, k, rho, p)
  simplified_equations = [simplify(eq) for eq in equations]

  for i, eq in enumerate(simplified_equations):
    print(f"Equation {i+1}: {eq}")