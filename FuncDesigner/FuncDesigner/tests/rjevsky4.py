from numpy import arange, array, ones, arange, zeros, cos
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 10
tau = 0.5
q = tau/n

x = oovar()

X = zeros(n)

m = max(x-X)

obj = ifThenElse(m>0, m, 0) -q*sum(x-X)

startPoint = {x: cos(arange(n))}

solvers = ['gsubg']

Colors = ['r', 'k','b']

for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 1700, name = 'rjevsky3 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.Ftol = 0.5e-2
    r = p.manage(solver, iprint=10, xtol = 1e-6, ftol = 1e-6, debug=0, show = solver == solvers[-1], plot = 0)
'''
--------------------------------------------------
solver: gsubg   problem: rjevsky3 (nVars: 10)    type: NSP   goal: minimum
 iter    objFunVal   
    0  9.789e-01 
   10  5.155e-03 
   12  2.168e-03 
istop: 16 (optimal solution wrt required Ftol has been obtained)
Solver:   Time Elapsed = 1.1 	CPU Time Elapsed = 1.1
objFunValue: 0.002167903
'''
