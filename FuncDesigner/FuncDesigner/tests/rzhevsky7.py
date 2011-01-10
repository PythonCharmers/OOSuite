from numpy import arange, array, ones,  zeros, cos, ceil, hstack, asarray
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

n = 30
x = oovars(n+2)

obj = sum([((0.0016+(x[k+1]-x[k])**2)/(0.04*(k+1)))**0.5 for k in range(n+1)])



#startPoint = {x: zeros(n+1)}
#startPoint = {x: cos(arange(n+1))}
X = [0]
for k in range(n+1):
    X.append(X[-1]+0.04*((0.0099099*(k+1))/(1-0.0099099*(k+1)))**0.5)
    
T_optPoint = dict([(x[i], X[i]) for i in range(n+2)])
startPoint = dict([(x[i], 0) for i in range(n+2)])
startPoint[x[-1]] = X[-1]

solvers = ['gsubg', 'ralg', 'amsg2p']
solvers = ['ralg', 'amsg2p']
def cb(p):
    tmp = ceil(log10(norm(asarray(X) - hstack((X[0], p.xk, X[-1])))))
    if tmp < cb.TMP:
#        print 'distance:', tmp, 'itn:', p.iter, 'n_func:', p.nEvals['f'], 'n_grad:', -p.nEvals['df']
        cb.TMP = tmp
        cb.stat['dist'].append(tmp)
        cb.stat['f'].append(p.nEvals['f'])
        cb.stat['df'].append(-p.nEvals['df'])
    return False
asa = lambda x:asarray(x).reshape(-1, 1)
Colors = ['r', 'k','b']
lines = []
R = {}

for i, solver in enumerate(solvers):
    p = NSP(obj, startPoint, fixedVars=(x[0], x[-1]), maxTime = 20, name = 'Rzhevsky7 (nVars: ' + str(n)+')', maxFunEvals=1e7, color = Colors[i])
    p._prepare()
    p.c=None
    #p.fEnough = 2.08983385058799+4e-10
    p.fOpt = obj(T_optPoint)
    p.fTol = 0.5e-15
    cb.TMP = 1000
    cb.stat = {'dist':[], 'f':[], 'df':[]}
    r = p.solve(solver, iprint=10, xtol = 1e-10, ftol = 1e-16, gtol = 1e-10, debug=0, show = solver == solvers[-1], plot = 0, callback = cb)
    print('objective evals: %d   gradient evals: %d ' % (r.evals['f'],r.evals['df']))
    print('distance to f*: %0.1e' % (r.ff-p.fOpt))
    print('distance to x*: %0.1e' % (norm(asarray(X) - hstack((X[0], p.xk, X[-1])))))
    R[solver] = hstack((asa(cb.stat['dist']), asa(cb.stat['f']), asa(cb.stat['df'])))
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
istop: 16 (optimal solution wrt required fTol has been obtained)
Solver:   Time Elapsed = 33.1 	CPU Time Elapsed = 29.28
objFunValue: 1.9171393 (feasible, max(residuals/requiredTolerances) = 0)
(theoretical:  2.08 - difference is due to unclear task description)
'''
