from numpy import arange, array, ones, zeros, cos, hstack, asarray, ceil
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 10
tau = 0.5
q = tau/n

x = oovar()

xOpt = X = 2 ** (-arange(1, n+1))

m = max(x-X)

obj = max((m, 0)) -q*sum(x-X)

startPoint = {x: X+1}

solvers = ['ralg','gsubg', 'amsg2p']
#solvers = ['ralg']

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



Colors = ['r', 'k','b']
R = {}
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 1700, name = 'Rzhevsky4 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
    p.fTol = 0.5e-20
    p.fOpt = 0.0
    cb.TMP = 1000
    cb.stat = {'dist':[], 'f':[], 'df':[]}
    r = p.solve(solver, iprint=10, xtol = 1e-20, ftol = 1e-20, debug=0, show = solver == solvers[-1], plot = 0, callback = cb)
    R[solver] = hstack((asa(cb.stat['dist']), asa(cb.stat['f']), asa(cb.stat['df'])))
'''
--------------------------------------------------
solver: gsubg   problem: rjevsky3 (nVars: 10)    type: NSP   goal: minimum
 iter    objFunVal   
    0  9.789e-01 
   10  5.155e-03 
   12  2.168e-03 
istop: 16 (optimal solution wrt required fTol has been obtained)
Solver:   Time Elapsed = 1.1 	CPU Time Elapsed = 1.1
objFunValue: 0.002167903
'''
