from openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros, sqrt, asscalar, inf
from pylab import legend, show, plot, subplot, xlabel, subplots_adjust
from string import rjust, ljust, expandtabs, center, lower
from FuncDesigner import *


N = 10
M = 5
Power = 1.13



x = oovar('x')

ff = (abs(x-M) ** Power).sum()
startPoint = {x:cos(arange(N))}

#c = [lambda x: 2* x[0] **4-32, lambda x: x[1]**2+x[2]**2 - 8]
c1 = 2* x[0] **4-32
c2 = x[1]**2+x[2]**2 - 8

#h1 = lambda x: 1e1*(x[-1]-1)**4
#h2 = lambda x: (x[-2]-1.5)**4

h1 = 1e1*(x[-1]-1)**4
h2 = (x[-2]-1.5)**4

#h = (h1, h2)



lb = -6*ones(N)
ub = 6*ones(N)
lb[3] = 5.5
ub[4] = 4.5

#lb = -1

gtol=1e-6
ftol = 1e-6
diffInt = 1e-8
contol = 1e-6
maxFunEvals = 1e6
maxTime = 10
Xlabel = 'time'
PLOT = 0
colors = ['b', 'k', 'y', 'g', 'r', 'm', 'c']

###############################################################
solvers = ['ralg', 'scipy_cobyla', 'lincher', 'scipy_slsqp', 'ipopt','algencan']
#solvers = ['ralg', 'ipopt']
solvers = ['ralg', 'scipy_cobyla', 'lincher', 'scipy_slsqp', 'ipopt']
solvers = ['ralg', 'scipy_slsqp', 'scipy_cobyla', 'algencan']
#solvers = ['ipopt','ralg', 'algencan']
solvers = ['ralg', 'scipy_cobyla']
#solvers = ['ralg', 'scipy_slsqp']
#solvers = ['ralg', 'algencan']
solvers = ['ralg']
###############################################################

lines, results = [], {}
for j, solver in enumerate(solvers):
    p = NLP(ff, startPoint, xlabel = Xlabel, gtol=gtol, diffInt = diffInt, ftol = ftol, maxIter = 1390, plot = PLOT, color = colors[j], iprint = 10, df_iter = 4, legend = solver, show=False,  contol = contol,  maxTime = maxTime,  maxFunEvals = maxFunEvals, name='NLP_bench_1')
    p.constraints = [c1<0,  c2<0,  h1.eq(0),  h2.eq(0), x > lb, x< ub]
    #p.constraints = h1.eq(0)
    
    #p._Prepare()
    #print p.dc(p.x0)
    #print h1.D(startPoint)
    #print h2.D(startPoint)
    #continue
    
    if solver =='algencan':
        p.gtol = 1e-2
    elif solver == 'ralg':
        pass
        #p.debug = 1
    
    #p.debug = 1
    
    r = p.solve(solver)
    for fn in ('h','c'):
        if not r.evals.has_key(fn): r.evals[fn]=0 # if no c or h are used in problem
    results[solver] = (r.ff, p.getMaxResidual(p.xk), r.elapsed['solver_time'], r.elapsed['solver_cputime'], r.evals['f'], r.evals['c'], r.evals['h'])
    if PLOT:
        subplot(2,1,1)
        F0 = ff(startPoint)
        lines.append(plot([0, 1e-15], [F0, F0], color= colors[j]))

if PLOT:
    for i in range(2):
        subplot(2,1,i+1)
        legend(lines, solvers)

    subplots_adjust(bottom=0.2, hspace=0.3)

    xl = ['Solver                              f_opt     MaxConstr   Time   CPUTime  fEvals  cEvals  hEvals']

    for i in range(len(results)):
        s=(ljust(lower(solvers[i]), 40-len(solvers[i]))+'%0.3f'% (results[solvers[i]][0]) + '        %0.1e' % (results[solvers[i]][1]) + ('      %0.2f'% (results[solvers[i]][2])) + '     %0.2f      '% (results[solvers[i]][3]) + str(results[solvers[i]][4]) + '   ' + rjust(str(results[solvers[i]][5]), 5) + ' '*8 +str(results[solvers[i]][6]))

        xl.append(s)

    xl = '\n'.join(xl)
    subplot(2,1,1)
    xlabel(Xlabel)

    from pylab import *
    subplot(2,1,2)
    xlabel(xl)
    show()

