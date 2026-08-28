from einsteinpy.symbolic import EinsteinTensor, MetricTensor
from sympy import symbols, sin, Function, solve
from sympy.printing.latex import LatexPrinter

# Constants
c, NEWTON_G, pi = symbols('c G \\pi')

# a printer that uses \dot and \ddot for time derivatives
class CustomLatexPrinter(LatexPrinter):
    def _print_Derivative(self, expr):
        if len(expr.args) == 2:
            func, der = expr.args
            var, order = der
            if var.is_Symbol and var.name == 't':
                if order == 1:
                    return r'\dot{' + func.name + '}(t)'
                elif order == 2:
                    return r'\ddot{' + func.name + '}(t)'
        return super()._print_Derivative(expr)

# Latex conversion function
def latex(expr):
    return CustomLatexPrinter().doprint(expr)

# Helper function to create diagonal matrices
def diag(*args):
    size = len(args)
    return [[args[i] if i == j else 0 for j in range(size)] for i in range(size)]

# Define the Friedmann metric class
class FriedmannMetric(MetricTensor):
    def __init__(self, coords, a, k, signature_sign=1):
        (t, r, theta, phi) = coords
        g = diag(c**2,
            -a**2/(1 - k*r**2),
            -a**2 * r**2,
            -a**2 * r**2 * sin(theta)**2)
        super().__init__(g, coords, name="Friedmann Metric")

def main():
    t, r, theta, phi, k = symbols('t r theta phi k')
    a = Function('a')(t)
    
    # The metric
    metric = FriedmannMetric((t, r, theta, phi), a, k)
    g = metric.tensor()
    print("g_{\\mu \\nu} = " + latex(g))
    print()
    
    # The Einstein tensor
    G = EinsteinTensor.from_metric(metric)
    G_comps = G.tensor()
    print("G_{\\mu \\nu} = " + latex(G_comps))
    print()
    
    # The energy-momentum tensor
    p, rho = symbols('p \\rho')
    T = diag(rho * c**4, -p * g[1][1], -p * g[2][2], -p * g[3][3])
    print("T_{\\mu \\nu} = " + latex(T))
    print()

    # The Einstein equations
    equations = [[G_comps[i][j] - 8 * pi * NEWTON_G / c**4 * T[i][j] for j in range(4)] for i in range(4)]
    none_vanishing_equations = [eq for row in equations for eq in row if eq != 0]
    for eq in none_vanishing_equations:
        print(latex(eq) + " = 0")
    print()

    # simplify and print the two independent equations
    eq1 = none_vanishing_equations[0].simplify(rational=True)
    eq2 = none_vanishing_equations[1].simplify(rational=True)
    for eq in [eq1, eq2]:
        print(latex(eq) + " = 0")
    print()

    # replacing \dot{a} and \ddot{a} with H and \dot{H}
    H = Function('H')(t)
    a_dot = H * a
    a_ddot = a * H**2 + a * H.diff(t)
    eq1_H = eq1.subs({a.diff(t): a_dot, a.diff(t, 2): a_ddot}).simplify()
    eq2_H = eq2.subs({a.diff(t): a_dot, a.diff(t, 2): a_ddot}).simplify()
    for eq in [eq1_H, eq2_H]:
        print(latex(eq) + " = 0")
    print()

    # isolate H^2 and \dot{H}
    sol = solve([eq1_H, eq2_H], (H**2, H.diff(t)))
    H_squared_eq = sol[H**2].simplify(rational=True)
    H_dot_eq = sol[H.diff(t)].simplify(rational=True)
    print("H^2 = " + latex(H_squared_eq))
    print("\\\\")
    print("dot{H} = " + latex(H_dot_eq))
    print()

    # Now use \dot{H} = \ddot{a}/a - H^2 to express \ddot{a}/a
    acceleration_eq = (H_dot_eq + H_squared_eq).factor()

    print("H^2 = " + latex(H_squared_eq))
    print("\\\\")
    print("\\frac{\\ddot{a}}{a} = " + latex(acceleration_eq))

if __name__ == "__main__":
    main()