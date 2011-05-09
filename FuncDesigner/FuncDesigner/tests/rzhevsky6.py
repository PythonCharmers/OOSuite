from numpy import arange, array, ones, zeros, cos, hstack, asarray, ceil
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 15
x = oovar()
X =  ones(n)
T = (0.01 * arange(1, 101))
tmp = []
for j in range(100):
    tmp.append(sum((x-X) * T[j]**arange(n))**2)
obj = sum(tmp)


startPoint = {x: 2*ones(n)}
def cb(p):
    tmp = ceil(log10(norm(1.0 - p.xk)))
    if tmp < cb.TMP:
#        print 'distance:', tmp, 'itn:', p.iter, 'n_func:', p.nEvals['f'], 'n_grad:', -p.nEvals['df']
        cb.TMP = tmp
        cb.stat['dist'].append(tmp)
        cb.stat['f'].append(p.nEvals['f'])
        cb.stat['df'].append(-p.nEvals['df'])
    return False
asa = lambda x:asarray(x).reshape(-1, 1)
solvers = ['ralg', 'amsg2p', 'gsubg']
solvers = [oosolver('amsg2p', gamma = 2.0)]
Colors = ['r', 'k','b']
lines = []
R = {}
for Tol_p in range(-10, -31, -1):
    #print('Tol = 10^%d' % Tol_p)
    for i, solver in enumerate(solvers):
        p = NSP(obj, startPoint, maxIter = 1700, name = 'Rzhevsky6 (nVars: ' + str(n)+')', maxTime = 300, maxFunEvals=1e7, color = Colors[i])
        p.fOpt = 0.0
        p.fTol = 10**Tol_p
        cb.TMP = 1000
        cb.stat = {'dist':[], 'f':[], 'df':[]}
        r = p.solve(solver, iprint=-1, xtol = 0, ftol = 0, callback = cb)
        R[solver] = hstack((asa(cb.stat['dist']), asa(cb.stat['f']), asa(cb.stat['df'])))
        print('itn df dx   %d   %0.1e   %0.1e' % (-r.evals['df'], r.ff-p.fOpt, norm(p.xk-1.0)))
#        print('objective evals: %d   gradient evals: %d ' % (r.evals['f'],r.evals['df']))
#        print('distance to f*: %0.1e' % (r.ff-p.fOpt))
#        print('distance to x*: %0.1e' % norm(p.xk-1.0))
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
