"""
This is test for future work on ralg
It may fail for now
"""


from FuncDesigner import *
from openopt import NLP
from numpy import nan

a, b = oovars('a', 'b')
f = a**2 + b**2

K = 1e5
minTreshold = 0.1

c1 = ifThenElse(a>minTreshold, K * a**2 + 1.0 / K * b**2, nan) < K * minTreshold**2
c2 = a>minTreshold


startPoint = {a:-K, b:-K}

p = NLP(f, startPoint, constraints = [c1,c2], iprint=10, maxIter = 1e4)

solver = 'ipopt'
solver = 'ralg'
#solver = 'scipy_slsqp'

r=p.solve(solver)
