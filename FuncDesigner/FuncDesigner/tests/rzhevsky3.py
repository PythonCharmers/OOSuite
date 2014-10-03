from numpy import arange, array, ones, zeros, cos, ceil,  hstack, asarray
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
obj = sum(sum(tmp))

startPoint = {x: zeros(n)}

solvers = ['ralg', 'amsg2p']
solvers = ['gsubg']

Colors = ['r', 'k','b']
xOpt = 1.0/n

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
lines = []
for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, maxIter = 4700, name = 'Rzhevsky3 (nVars: ' + str(n)+')', maxTime = 30000, maxFunEvals=1e7, color = Colors[i])
    #p.maxIter = 10#; p.useSparse = False
    p.fEnough = 1.0e-5
    p.fOpt = 1.0e-5
    p.fTol = 0.5e-5
    cb.TMP = 1000
    cb.stat = {'dist':[], 'f':[], 'df':[]}
    r = p.manage(solver, iprint=1, xtol = 1e-10, ftol = 1e-10, show = solver == solvers[-1], plot = 0, callback = cb)
    R[solver] = hstack((asa(cb.stat['dist']), asa(cb.stat['f']), asa(cb.stat['df'])))
