from FuncDesigner import *
from openopt import NLP

a, b, c = oovars('a', 'b', 'c')
f = sum(a*[1, 2])**2+b**2+c**2
startPoint = {a:[100, 12], b:2, c:40} # however, you'd better use numpy arrays instead of Python lists
p = NLP(f, startPoint)
p.constraints = [(2*c+a-10)**2<1.5, (a-10)**2<1.5, a[0]>8.9, a+b>[ 7.97999836, 7.8552538 ], \
a<9, (c-2)**2<1, b<-1.02, c>1.01, ((b+2*c-1)**2).eq(0)]
r = p.solve('ralg', plot=True)
print r.xf

""" Expected output:
solver: ralg problem: unnamed goal: minimum
iter objFunVal log10(maxResidual)
0 1.698e+04 4.46
...
72 7.176e+02 -6.21
istop: 3 (|| X[k] - X[k-1] || < xtol)
Solver: Time Elapsed = 1.37 CPU Time Elapsed = 1.01
objFunValue: 717.64993 (feasible, max constraint = 6.12135e-07)
{a: array([ 8.9999972 , 8.87525296]), b: array([-1.01999945]), c: array([ 1.01000767])} 
"""
