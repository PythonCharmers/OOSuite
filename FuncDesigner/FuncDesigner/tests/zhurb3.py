from numpy import arange
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

N = 100000
x = oovar('x')
startPoint = {x: 1 + 1.0 / arange(1, N+1)}
S = 1e4 ** (1.0/arange(1, N+1))

arr = sin(arange(N))
f = sum((x-arr)**2 * S) / 1e4


solvers = [oosolver('ralg')]
solvers = [oosolver('gsubg', dual=True, zhurb = 50)]
#solvers = [oosolver('gsubg', zhurb = 20, dual=False)]
#solvers = ['ipopt']
#solvers = ['slmvm2']
#solvers = ['mma']
for solver in solvers:
    p = NSP(f, startPoint, maxIter = 10000, maxTime = 15000, maxFunEvals=1e7)
    p.fEnough = 1.5e-1
    p.fTol = 5.0e-1
    #p.constraints = (y > 5)(tol=1e-4) #x>1e-1 #[2*y<sin(arange(N))]
    #r = p.solve(solver, iprint=10, xtol = 1e-36, ftol = 1e-16, show = solver == solvers[-1])
    r = p.manage(solver, iprint=1, xtol = 1e-8, ftol = 1e-7, show = solver == solvers[-1])
    

