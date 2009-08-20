



"""
Some non-linear functions have much more restricted dom than R^nVars.
For example F(x) = log(x); dom F = R+ = {x: x>0}

For optimization solvers it is wont to expect user-povided F(x) = nan if x is out of dom.

I can't inform how successfully OO-connected solvers
will handle a prob instance with restricted dom
because it seems to be too prob-specific

Still I can inform that ralg handles the problems rather well
provided in every point x from R^nVars at least one ineq constraint is active
(i.e. value constr[i](x) belongs to R+)

Note also that some solvers require x0 inside dom objFunc.
For ralg it doesn't matter.
"""

from numpy import *
from openopt import NLP

n = 10
an = arange(n) # array [0, 1, 2, ..., n-1]
x0 = n+15*(1+cos(an))

# from all OO-connected NLP solvers
# only ralg can handle x0 out of dom objFunc:
# x0 = n+15*(cos(an))

f = lambda x: (x**2).sum() + sqrt(x**3-arange(n)**3).sum()
df = lambda x: 2*x + 0.5*3*x**2/sqrt(x**3-arange(n)**3)
c = lambda x: an**3 - x**3
dc = lambda x: diag(-3 * x**2)

# you can use splitting of constraints, for some solvers it yields speedup:
#c, dc = [], []
#for i in xrange(n):
#    c += [lambda x, i=i: i**3-x[i]**3]
#    dc += [lambda x, i=i: hstack((zeros(i), -3*x[i]**2, zeros(n-i-1)))]

lb = arange(n)
solvers = ['ralg', 'scipy_slsqp', 'scipy_cobyla', 'ipopt',  'algencan']
for solver in solvers:
    p = NLP(f, x0, df=df, lb=lb, c=c, dc=dc, iprint = 100, maxIter = 10000, maxFunEvals = 1e8)
    #p.checkdf()
    #p.checkdc()
    r = p.solve(solver)
# expected r.xf = [0, 1, 2, ..., n-1]
