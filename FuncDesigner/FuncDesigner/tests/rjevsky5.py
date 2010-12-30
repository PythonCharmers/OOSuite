from numpy import arange, array, ones, arange, zeros, cos
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 50
x = oovar()
A = zeros((n, n))
b = zeros(n)
for i in range(n):
    for j in range(n):
        A[i, j] = 1.0 / (i+j+1) # Python indexation from zero
    b[i] = sum(A[i, :])
        
#startPoint = {x: cos(arange((n)))}
startPoint = {x: 2*ones(n)}
obj = dot(x, dot(A, x)) / 2 - dot(b, x)

#solvers = [gsubg, 'ralg', 'ipopt']
solvers = ['ralg', 'amsg2p']

Colors = ['r', 'k','b']

lines = []
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 1700, name = 'rjevsky5 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.fTol = 0.5e-7
    p.fEnough = p.fOpt = -34.4086089
    r = p.manage(solver, iprint=1, xtol = 1e-6, ftol = 1e-9, debug=0, show = solver == solvers[-1], plot = 0)
'''
solver: gsubg   problem: rjevsky5 (nVars: 50)    type: NSP   goal: minimum
 iter    objFunVal   
    0  -2.241e+00 
    1  -3.056e+01 
    2  -3.387e+01 
    3  -3.427e+01 
    4  -3.434e+01 
    5  -3.436e+01 
    6  -3.439e+01 
    7  -3.439e+01 
    8  -3.439e+01 
    9  -3.440e+01 
   10  -3.440e+01 
   11  -3.440e+01 
   12  -3.441e+01 
   13  -3.441e+01 
   14  -3.441e+01 
   15  -3.441e+01 
   16  -3.441e+01 
   17  -3.441e+01 
   18  -3.441e+01 
   19  -3.441e+01 
   20  -3.441e+01 
   21  -3.441e+01 
istop: 16 (optimal solution wrt required fTol has been obtained)
Solver:   Time Elapsed = 1.01 	CPU Time Elapsed = 0.95
objFunValue: -34.408462
(theoretical:  -34.40860897)
'''
