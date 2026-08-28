from einsteinpy.symbolic import EinsteinTensor, RiemannCurvatureTensor, MetricTensor
from sympy import symbols, sin, sinh, Function, solve
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
class FLRWMetric(MetricTensor):
    def __init__(self, coords, a):
        (t, chi, theta, phi) = coords
        g = diag(c**2,
            -a**2,
            -a**2 * sinh(chi)**2,
            -a**2 * sinh(chi)**2 * sin(theta)**2)
        super().__init__(g, coords, name="Friedmann Metric")

class MonkowskiMetric(FLRWMetric):
    def __init__(self, coords):
        super().__init__(coords, 1)

def main():
    t, chi, theta, phi, k = symbols('t chi theta phi k')
    a = c * t
    
    # The metric
    metric = FLRWMetric((t, chi, theta, phi), a)
    g = metric.tensor()
    print("g_{\\mu \\nu} = " + latex(g))
    print()

    # The Riemann tensor
    R = RiemannCurvatureTensor.from_metric(metric)
    R_comps = R.tensor()
    print("R_{\\mu \\nu} = " + latex(R_comps))
    print()

if __name__ == "__main__":
    main()