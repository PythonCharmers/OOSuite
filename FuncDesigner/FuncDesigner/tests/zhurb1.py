from numpy import arange, array
from numpy.linalg import norm
from openopt import NSP, oosolver
from FuncDesigner import *

K = 20
N = 1000

#K = 2
#N = 1

x, y = oovars('x', 'y')
a = 10*sin(arange(1, N+1))
b = arange(1, N+1)
#f1 = sum((y+5)**2 * sqrt(arange(1, N+1))) / N
f1 = sum((y+a)**2 * b) / (10**4 *N) 
#f1 = (y)*2
#startPoint = {y:cos(arange(N))}
#----------------------------

f2 = abs(x+10)**2 
startPoint = {x:1, y:1}

# or
f2 = (abs(x[0]+5*x[1]-0.2) + K * abs(x[1]-5*x[2]+0.3) + 2 * K**2 * abs(x[2]+5*x[1]-0.05))/100
startPoint = {x:[10, 0.3, 0.2], y:cos(arange(N))}

# or
f2 = abs(x[0]) + sum(abs(x[1:K]-100*x[0:K-1]) * array(2)**arange(K-1))/K**5
startPoint = {x:cos(arange(K)), y:cos(arange(N))}

# or
#f2 = Max(abs(x))
#startPoint = {x:100 + cos(arange(10)), y:cos(arange(N))}

# or

#f2 = Max([5*abs(x[0])+1, 50*abs(x[1])+2, 500*x[2]**2+4])
#startPoint = {x:100 + cos(arange(3)), y:cos(arange(N))}


#----------------------------

f = f1 + f2 



print 'start point: f1 = %e   f2 = %e' % (f1(startPoint), f2(startPoint))
#print "start point: norm(f1') = %e   norm(f2') = %e" % (norm(f1.D(startPoint, y)), norm(f2.D(startPoint, x)))

ralg = oosolver('ralg')
gsubg = oosolver('gsubg', addASG = False)

solvers = [ralg]

#solvers = [oosolver('gsubg', zhurb = 20, dual=False)]
solvers = ['ipopt', gsubg, 'scipy_cg']
solvers = [gsubg]
#solvers = ['ipopt']
#solvers = ['slmvm2']
#solvers = ['slmvm1']
#solvers = ['mma']
Colors = ['r', 'k','b']

lines = []
for i, solver in enumerate(solvers):
    p = NSP(f, startPoint, maxIter = 300, name = 'ns' + str(N+K), maxTime = 15000, maxFunEvals=1e7, color = Colors[i])
    #p.constraints = y>-100
    p.fEnough = 1.0e-1#e-1
    p.fTol = 0.5e-1
    p.debug = 1
    #p.constraints = (y > 5)(tol=1e-4) #x>1e-1 #[2*y<sin(arange(N))]
    #r = p.manage(solver, iprint=10, xtol = 1e-9, ftol = 1e-9, show = solver == solvers[-1], maxIter = 10000)
    r = p.solve(solver, iprint=10, xtol = 1e-6, ftol = 1e-6, debug=0, show = solver == solvers[-1], plot = 0)
    print 'end point: f1 = %e   f2 = %e' % (f1(r.xf), f2(r.xf))
