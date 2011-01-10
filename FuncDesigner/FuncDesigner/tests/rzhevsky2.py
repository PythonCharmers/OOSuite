from numpy import arange, array, ones, zeros, cos, ceil, hstack, asarray
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 10

x = oovar('x')
y = oovar('y')
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

#f1 = ifThenElse(phi[0]>phi[1], phi[0], phi[1])
#f2 = ifThenElse(phi[2]>f1, phi[2], f1)
#f3 = ifThenElse(phi[3]>f2, phi[3], f2)
#f4 = ifThenElse(phi[4]>f3, phi[4], f3)
#
#obj = y + ifThenElse(f4-y>0, f4-y, 0)

obj = y + max((0, max(phi)-y))

startPoint = {x: ones(n), y:1}

solvers = ['ralg', 'amsg2p', 'gsubg']
Colors = ['r', 'k','b']

xOpt = [-1.26257e-1, -3.43781e-2, -6.85720e-3, 2.63607e-2, 6.72950e-2, -2.78399e-1, 7.42187e-2, 1.38524e-1, 8.40313e-2, 3.85803e-2, -0.841408334596]

def cb(p):
    tmp = ceil(log10(norm(xOpt - p.xk)))
    if tmp < cb.TMP:
#        print 'distance:', tmp, 'itn:', p.iter, 'n_func:', p.nEvals['f'], 'n_grad:', -p.nEvals['df']
        cb.TMP = tmp
        cb.stat['dist'].append(tmp)
        cb.stat['f'].append(p.nEvals['f'])
        cb.stat['df'].append(-p.nEvals['df'])
    return False
asa = lambda x:asarray(x).reshape(-1, 1)

R = {}
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 17000, name = 'Rzhevsky2 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.fTol = 0.5e-12
    p.maxTime = 100
    #p.fEnough = -0.841408334595
    p.fOpt = -0.841408334596
    cb.TMP = 1000
    cb.stat = {'dist':[], 'f':[], 'df':[]}
    r = p.solve(solver, iprint=10, ftol = 1e-15, xtol = 1e-12, show = solver == solvers[-1], plot = 0, callback = cb)
    R[solver] = hstack((asa(cb.stat['dist']), asa(cb.stat['f']), asa(cb.stat['df'])))
'''
--------------------------------------------------
solver: gsubg   problem: rjevsky2 (nVars: 10)    type: NSP   goal: minimum
 iter    objFunVal   
    0  5.337e+03 
   10  -5.386e-01 
   20  -8.399e-01 
   30  -8.414e-01 
   40  -8.414e-01 
   50  -8.414e-01 
   53  -8.414e-01 
istop: 16 (optimal solution wrt required fTol has been obtained)
Solver:   Time Elapsed = 4.55 	CPU Time Elapsed = 4.54
objFunValue: -0.84138193
(theoretical:  -0.841408334596)
'''
