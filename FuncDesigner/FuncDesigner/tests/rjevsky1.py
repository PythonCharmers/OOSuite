from numpy import arange, array, ones, arange, zeros, cos
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 10

x = oovar('x')
A = [zeros((n, n)) for k in range(5)]
b = [zeros(n) for k in range(5)]
s = zeros((n, n))
for k in range(5):
    for j in range(n):
        for i in range(j):
            A[k][i, j] = exp(float(i+1)/(j+1)) * cos((i+1)*(j+1)) *sin(k+1)
    A[k] = A[k] + A[k].T 

for k in range(5):
    for i in range(n):
        s[k, i] = sum(abs(A[k][i, :]))

for k in range(5):
    for i in range(n):
        b[k][i] = exp(float(i+1)/(k+1)) * sin((i+1)*(k+1))
        A[k][i, i] = (i+1)*abs(sin(k+1)) / 10.0 + s[k, i]#sum(A[k][i, :])-A[k][i, i] 

phi = [dot(x, dot(A[k], x)) - dot(b[k], x) for k in range(5)]

f1 = ifThenElse(phi[0]>phi[1], phi[0], phi[1])
f2 = ifThenElse(phi[2]>f1, phi[2], f1)
f3 = ifThenElse(phi[3]>f2, phi[3], f2)
f4 = ifThenElse(phi[4]>f3, phi[4], f3)

obj = f4
startPoint = {x: cos(arange(n))}

solvers = ['gsubg']
Colors = ['r', 'k','b']

for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 17000, name = 'rjevsky1 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.Ftol = 0.5e-2
    r = p.solve(solver, iprint=10, xtol = 1e-16, ftol = 1e-36, gtol = 1e-200, debug=0, show = solver == solvers[-1], plot = 0)
'''
--------------------------------------------------
solver: gsubg   problem: rjevsky1 (nVars: 10)    type: NSP   goal: minimum
 iter    objFunVal   
    0  1.573e+02 
   10  -7.780e-01 
   20  -8.410e-01 
   30  -8.410e-01 
   40  -8.410e-01 
   50  -8.412e-01 
   60  -8.413e-01 
   70  -8.413e-01 
   80  -8.413e-01 
   90  -8.413e-01 
  100  -8.413e-01 
  102  -8.413e-01 
istop: 17 (all new subgradients have been removed due to the angle threshold)
Solver:   Time Elapsed = 5.56 	CPU Time Elapsed = 5.56
objFunValue: -0.84128185
(theoretical:  -0.841408334596)
'''
