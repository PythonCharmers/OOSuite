from numpy import arange, array, ones, arange, zeros, cos
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 15
x = oovar()
X =  ones(n)
T = (0.01 * arange(1, 101))
tmp = []
for j in range(100):
    tmp.append(((x-X) * T[j]**arange(n))**2)
obj = sum(tmp)

#startPoint = {x: zeros(n)}
startPoint = {x: 2*ones(n)}

#solvers = [gsubg, 'ralg', 'ipopt']
solvers = ['gsubg']

Colors = ['r', 'k','b']

lines = []
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 1700, name = 'rjevsky6 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.fEnough = 1.0e-2
    p.Ftol = 0.5e-2
    r = p.manage(solver, iprint=1, xtol = 1e-6, ftol = 1e-6, debug=0, show = solver == solvers[-1], plot = 0)
'''
--------------------------------------------------
solver: gsubg   problem: rjevsky6 (nVars: 15)    type: NSP   goal: minimum
 iter    objFunVal   
    0  2.408e+02 
    1  9.712e+01 
    2  5.315e+01 
    3  3.569e+01 
    4  1.432e+01 
    5  8.967e+00 
    6  4.438e+00 
    7  1.715e+00 
    8  1.157e+00 
    9  4.905e-01 
   10  3.594e-01 
   11  1.946e-01 
   12  1.026e-01 
   13  6.970e-02 
   14  3.963e-02 
   15  1.357e-02 
   16  9.511e-03 
   17  9.511e-03 
istop: 10 (fEnough has been reached)
Solver:   Time Elapsed = 63.66 	CPU Time Elapsed = 62.98
objFunValue: 0.0095110214
(theoretical:  0)
'''
