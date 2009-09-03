from openopt import NLP
from FuncDesigner import *


a, b, c = oovars('a', 'b', 'c')
f = sum(a*[1, 2])**2+b**2+c**2
startPoint = {a:[100, 12], b:2, c:40} # however, you'd better use numpy arrays instead of Python lists
p = NLP(f, startPoint)
p.constraints = [(2*c+a-10)**2<1.5 + 0.1*b, (a-10)**2<1.5, a[0]>8.9, a+b>[ 7.97999836, 7.8552538 ], \
a<9, (c-2)**2<1, b<-1.02, c>1.01, ((b+2*c-1)**2).eq(0)]
r = p.solve('ralg', plot=False)
print r.xf

""" Expected output:
...
objFunValue: 717.65008 (feasible, max constraint =  5.57207e-07)
{a: array([ 9.00000015,  8.87525291]), b: array([-1.01999967]), c: array([ 1.01000454])}
"""
