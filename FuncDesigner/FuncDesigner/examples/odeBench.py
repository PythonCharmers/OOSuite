from time import time
from numpy import linspace, pi, hstack
from FuncDesigner import *

sigma = 1e-7 # interalg works even with 1e-26, i.e. 10^-26
StartTime, EndTime = 0, 10
times = linspace(StartTime, EndTime, 100) # 0, 0.01, 0.02, 0.03, ..., 10

# required accuracy
# I use so big value for good graphical visualization below, elseware 2 lines are almost same and difficult to view
ftol = 0.05 # this value is used by interalg only, not by scipy_lsoda

t = oovar()
f = exp(-(t-4.321)**2/(2*sigma)) / sqrt(2*pi*sigma) + 0.1*sin(t)
# optional, for graphic visualisation and exact residual calculation:
from scipy.special import erf
exact_sol = lambda t: 0.5*erf((t-4.321)/sigma) - 0.1*cos(t) # + const, that is a function from y0

results = {}
for solver in ('scipy_lsoda', 'interalg'):
    y = oovar()
    equations = {y: f} # i.e. dy/dt = f
    startPoint = {y: 0} # y(t=0) = 0
    # assign ODE. 3rd argument (here "t") is time variable that is involved in differentiation
    myODE = ode(equations, startPoint, t, times, ftol = ftol)# 
    T = time()
    r = myODE.solve(solver, iprint = -1)
    print('%s ODE time elapsed: % f' % (solver,  time()-T))
    Y = r(y)
    results[solver] = Y
    print('%s result in final time point: %f' % (solver, Y[-1]))
'''
scipy_lsoda ODE time elapsed:  0.044204
scipy_lsoda result in final time point: 0.183907
interalg ODE time elapsed:  0.110030 (here most of time was elapsed for spline interpolation of obtained results)
interalg result in final time point: 1.184044
'''

realSolution = exact_sol(times) - exact_sol(times[0]) + startPoint[y] 
print('max scipy.interpolate.odeint difference from real solution: %0.9f' \
      % max(abs(realSolution - results['scipy_lsoda'])))
print('max interalg difference from real solution: %0.9f (required: %0.9f)' \
      % (max(abs(realSolution - results['interalg'])), ftol))
'''
max scipy.interpolate.odeint difference from real solution: 1.000000020
max interalg difference from real solution: 0.025937095 (required: 0.050000000)
'''
# Now let's see a graphical visualization of results
from pylab import hist, show, plot, grid, legend, title
#hist(Times, 500)

p1,  = plot(times, results['interalg'], 'b')
p2,  = plot(times, results['scipy_lsoda'], 'r')
p3,  = plot(times, realSolution,'k')
legend([p1, p2, p3], ['interalg', 'scipy.interpolate.odeint', 'exact solution'], 'best')
grid('on')
show()
