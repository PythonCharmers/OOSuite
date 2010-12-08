from numpy import arange, array, ones, arange, zeros, cos
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 30
x = oovar()

obj = sum([((0.0016+(x[k+1]-x[k])**2)/(0.04*(k+1)))**0.5 for k in range(n)])



#startPoint = {x: zeros(n)}
startPoint = {x: cos(arange(n+1))}

#solvers = [gsubg, 'ralg', 'ipopt']
solvers = ['gsubg']

Colors = ['r', 'k','b']

lines = []
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 1700, name = 'rjevsky6 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.fEnough = 1.0e-2
    p.Ftol = 0.5e-2
    r = p.manage(solver, iprint=10, xtol = 1e-6, ftol = 1e-6, debug=0, show = solver == solvers[-1], plot = 0)
'''
--------------------------------------------------
solver: gsubg   problem: rjevsky6 (nVars: 30)    type: NSP   goal: minimum
 iter    objFunVal    log10(MaxResidual/ConTol)   
    0  2.954e+01                   -100.00 
OpenOpt Warning: Handling of constraints is not implemented properly for the solver gsubg yet
   10  4.002e+00                   -100.00 
   20  2.232e+00                   -100.00 
   30  1.975e+00                   -100.00 
   40  1.943e+00                   -100.00 
   50  1.927e+00                   -100.00 
   60  1.921e+00                   -100.00 
   70  1.918e+00                   -100.00 
   80  1.917e+00                   -100.00 
   90  1.917e+00                   -100.00 
  100  1.917e+00                   -100.00 
  110  1.917e+00                   -100.00 
istop: 16 (optimal solution wrt required Ftol has been obtained)
Solver:   Time Elapsed = 33.1 	CPU Time Elapsed = 29.28
objFunValue: 1.9171393 (feasible, max(residuals/requiredTolerances) = 0)
(theoretical:  2.08 - difference is due to unclear task description)
'''
