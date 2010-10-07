from numpy import arange
from openopt import NSP, oosolver
from FuncDesigner import *

K = 1000
N = 15000


x, y = oovars(2)

f1 = sum(y**2 * (arange(1, N+1)))
f2 = K*abs(x[0]+5*x[1]-0.2) + K**2 * abs(x[1]-5*x[2]+0.3) + 2 * K**3 * abs(x[2]+5*x[1]-0.05)
f = (f1 + f2) / N

#f = sum(abs(y)) + abs(x[0]) + abs(x[1])*K + abs(x[2]) * K**2

startPoint = {x:[10, 0.011, 0.01], y:cos(arange(N))}
#s = f.D(startPoint)
#raise 0
solvers = [oosolver('ralg', zhurb=20), 'ralg']
solvers = [oosolver('ralg', zhurb=20)]
#solvers = ['ipopt']
#solvers = ['slmvm2']
#solvers = ['mma']
for solver in solvers:
    p = NSP(f, startPoint, maxIter = 10000, maxTime = 15000, maxFunEvals=1e7)
    p.fEnough = 5e-9
    #p.constraints = [2*y<sin(arange(N))]
    r = p.solve(solver, iprint=10, show = solver == solvers[-1])

