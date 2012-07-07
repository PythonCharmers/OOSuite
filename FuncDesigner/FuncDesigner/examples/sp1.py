from FuncDesigner import *
from openopt import NLP
 
A = distribution.gauss(4, 0.5) # gauss distribution with mean = 4, std = 0.5
# this is same to
#from scipy import stats
#_a = distribution.continuous(ppf=stats.norm(4, 0.5).ppf)
# along with "gauss" you can use "norm" (for scipy.stats compatibility, yet I dislike it due to ambiguity with linalg.norm)
# or "normal"
 
B = distribution.exponential(3, 0.7)
# for compatibility with scipy.stats you can use "expon"
 
C = distribution.uniform(-1.5, 1.5) # uniform distribution from -1.5 to 1.5
 
a, b, c = oovars('a b c')
x, y, z = oovars('x y z', lb=-1, ub=1)
y = z = 0
f = x*a**2 + y*b**2 + z*c**2 + (x-1)**2 + y**2 + (z-5)**2
objective = 0.15 * mean(f+2*x) - 20 * z * var(b*exp(x))  + 15*y**2 * var(c*exp(z)) 
constraints = [
               mean(b+y) <= 3.7, 
               std(x*a+z) < 0.4 
               ]
startPoint = {x: 0,  a: A, b: B, c: C}
 
p = NLP(objective, startPoint)#, constraints = constraints, freeVars = x)
# select gradient-free solver
solver = 'goldenSection' # for unconstrained and box-bounded problems you'd better use bobyqa
# see http://openopt.org/NLP for more available solvers
r = p.maximize(solver, iprint = 1, maxDistributionSize=700, plot = False)
#"plot = True" means real-time graphics output of convergence, requires matplotlib installed
''' Results for Intel Atom 1.6 GHz:
------------------------- OpenOpt 0.39 -------------------------
solver: scipy_cobyla   problem: unnamed    type: NLP   goal: maximum
 iter   objFunVal   log10(maxResidual)   
    0  3.900e+00            -100.00 
    1  1.155e+01              -0.00 
    2  6.284e+01              -1.47 
    3  5.176e+01              -3.58 
    4  5.315e+01              -3.13 
    5  5.313e+01              -4.09 
    6  5.313e+01            -100.00 
    7  5.313e+01              -6.27 
    8  5.313e+01            -100.00 
    9  5.313e+01            -100.00 
istop: 1000
Solver:   Time Elapsed = 24.82 	CPU Time Elapsed = 24.57
objFunValue: 53.125519 (feasible, MaxResidual = 0)
'''
print(r(x, y, z)) # [0.81121226345864939, 0.0087499573476878291, -1.0]
