from FuncDesigner import *
from openopt import NLP
from numpy import arange
a, b = oovars('a', 'b')

N = 100000

#def CostlyFunction(z):
#    r = 0.0
#    for k in xrange(1, K+2):
#        r += z ** (1 / k**1.5)
#    return r
    
startPoint = {a:1, b: cos(arange(N))} # however, you'd better use numpy arrays instead of Python lists

ff = sum(exp(b)*cos(b)/exp(b)/cos(b))('ff')
objective = a**2 + ff

# Optional: set come constraints
constraints = []
#constraints = [a>[1, 1], # it means a[0]>=1, a[1]>=1
#               (a.sum()+b)**2<10, 
#               (a+b+c)**2 < 15] # it means (a[i]+b+c)**2 < 15 for  i in {0,1}

fixedVars = b # or fixedVars = [b] / c / [b,c] / [a,c] etc
p = NLP(objective, startPoint, fixedVars = fixedVars, constraints = constraints)

command = """
r = p.solve('ralg')
"""
import cProfile
cProfile.runctx( command, globals(), locals(), filename="optFixedVars.profile" )

print 'ff same: ', ff.same, 'ff evals: ', ff.evals
print 'ff.D same: ', ff.same_d, 'ff.D evals: ', ff.evals_d

print('opt_a: ', a(r))
#
## Alternatively, you can set optVars instead of fixedVars: 
#optVars = [a, c] # or optVars = [a] / c / [b,c] / [a,c] etc
#p = NLP(objective, startPoint, optVars = optVars, constraints = constraints)
#r = p.solve('ralg')
#print('opt_a: ', a(r), '   opt_c: ', c(r))
#
#"""
#Expected output:
#...
#objFunValue: 5.8140327 (feasible, MaxResidual = 5.79053e-07)
#('opt_a: ', array([ 1.00000026,  0.99999942]), '   opt_c: ', array([ 0.06353573]))
#"""
