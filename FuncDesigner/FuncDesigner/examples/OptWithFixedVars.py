from FuncDesigner import *
from openopt import NLP

a, b, c = oovars('a', 'b', 'c')
startPoint = {a:[100, 12], b:-0.9, c:40} # however, you'd better use numpy arrays instead of Python lists

objective = ((a*[1, 2])**2).sum() + b**2 + c**2

# Optional: set come constraints
constraints = [a>[1, 1], (b+a)**2<10, (a+b+c)**2 < 15]

fixedVars = b
# or fixedVars = [b] / c / [b,c] / [a,c] etc
p = NLP(objective, startPoint, fixedVars = fixedVars, constraints =constraints)

r = p.solve('ralg', storeIterPoints=1)
print r.xf

# Alternatively, you can set optVars instead of fixedVars: 
optVars = [a, c]
# or optVars = [a] / c / [b,c] / [a,c] etc
p = NLP(objective, startPoint, optVars = optVars, constraints =constraints)
r = p.solve('ralg')
print r.xf

"""
Expected output:
...
objFunValue: 5.8100177 (feasible, max constraint =  6.6959e-07)
{b: -0.90000000000000002, a: array([ 1.00000905,  0.99999933]), c: array([-0.00221527])}
"""
