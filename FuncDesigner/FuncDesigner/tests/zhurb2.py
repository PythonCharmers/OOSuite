from numpy import arange
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

N = 300
x = oovar('x')
startPoint = {x: 1 + 1.0 / arange(1, N)}
S = 1e4 ** (1.0/arange(1, N))

#f = abs(x[0]) + S * abs(x[1]) + S**2 * abs(x[2])
f = sum(abs(x)*S)


solvers = [oosolver('ralg')]
solvers = [oosolver('gsubg', addASG = True)]
#solvers = [oosolver('gsubg', zhurb = 20, dual=False)]
#solvers = ['ipopt']
#solvers = ['slmvm2']
#solvers = ['mma']
for solver in solvers:
    p = NSP(f, startPoint, maxIter = 10000, maxTime = 15000, maxFunEvals=1e7)
    p.fEnough = 1.5e1
    p.fTol = 1.0e1
    #p.constraints = (y > 5)(tol=1e-4) #x>1e-1 #[2*y<sin(arange(N))]
    #r = p.solve(solver, iprint=10, xtol = 1e-36, ftol = 1e-16, show = solver == solvers[-1])
    r = p.solve(solver, iprint=10, xtol = 1e-16, ftol = 1e-6, show = solver == solvers[-1])
    

