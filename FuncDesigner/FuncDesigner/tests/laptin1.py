from FuncDesigner import *
from openopt import NLP
from numpy import arange, ones

n = 50
kappa = 1.05
Tol = 1e+16

c = arange(1,n+1)/10.0

x = oovar('x')
startPoint = {x:ones(n)}

xs = sum(x)

cons = (x  <= kappa/n * xs)(tol=-1e-6)

f = sum(c*x) + 1e-300*sum(sqrt(-x + kappa/n * xs + Tol)) 

p = NLP(f, startPoint, constraints=cons)

r = p.minimize('gsubg', dilation = False, iprint = 10, ftol=1e-10, fTol = 1e-4, xtol = 1e-6, maxIter = 1e5, maxFunEvals=1e7, T='float128')
print 'objective func evaluations: ', r.evals['f']
