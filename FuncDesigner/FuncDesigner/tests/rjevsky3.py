from numpy import arange, array, ones, arange, zeros, cos
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 20
x = oovar()
X = 1.0 / n * ones(n)
T = (0.01 * arange(1, 101))
tmp = []
for j in range(100):
    tmp.append(abs((x-X) * T[j]**arange(n)))
obj = sum(tmp)

startPoint = {x: zeros(n)}
#startPoint = {x: cos(arange(n))/n}

#solvers = [gsubg, 'ralg', 'ipopt']
solvers = ['gsubg']

Colors = ['r', 'k','b']

lines = []
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 1700, name = 'rjevsky3 (nVars: ' + str(n)+')', maxTime = 3000, maxFunEvals=1e7, color = Colors[i])
    p.fEnough = 1.0e-5
    p.Ftol = 0.5e-5
    r = p.manage(solver, iprint=1, xtol = 1e-10, ftol = 1e-10, debug=0, show = solver == solvers[-1], plot = 0)
