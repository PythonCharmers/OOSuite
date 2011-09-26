from time import time
from numpy import linspace, pi, hstack
from FuncDesigner import *

sigma = 1e-15
StartTime, EndTime = 0, 10
times = linspace(StartTime, EndTime, 1000) # 1000 points between StartTime, EndTime

# required accuracy
# I use so big value for good graphical visualization below, elseware 2 lines are almost same and difficult to view
ftol = 0.00005 # this value is used by interalg only, not by scipy_lsoda

t = oovar()
f = exp(-(t-4.321)**2/(2*sigma)) / sqrt(2*pi*sigma) + 0.1*sin(t)
# optional, for graphic visualisation and exact residual calculation:
from scipy.special import erf
exact_sol = lambda t: 0.5*erf((t-4.321)/sigma) - 0.1*cos(t) # + const, that is a function from y0

y = oovar()
equations = {y: f} # i.e. dy/dt = f
startPoint = {y: 0} # y(t=0) = 0

# assign ODE. 3rd argument (here "t") is time variable that is involved in differentiation
myODE = ode(equations, startPoint, t, times, ftol = ftol)
T = time()
r = myODE.solve('interalg', iprint = -1)
print('Time elapsed with user-defined solution time intervals: %0.1f' % (time()-T))
Y = r(y)
print('result in final time point: %f sec' % Y[-1])


# now let interalg choose time points by itself
# we provide 4th argument as only 2 time points (startTime, endTime)
myODE = ode(equations, startPoint, t, (times[0], times[-1]), ftol = ftol)# 
T = time()
r = myODE.solve('interalg', iprint = -1)
print('Time elapsed with automatic solution time intervals: %0.1f' % (time()-T))
Y = r(y)
print('result in final time point: %f sec' % Y[-1])

'''
Time elapsed with user-defined solution time intervals: 11.2
result in final time point: 1.183941 sec
Time elapsed with automatic solution time intervals: 3.1
result in final time point: 1.183908 sec

In time autoselect mode r(t) and r(y) will be arrays of times and values:
>>> r(t)
array([  1.90734863e-05,   5.72204590e-05,   9.53674316e-05, ...,
         9.99990463e+00,   9.99994278e+00,   9.99998093e+00])
>>> r(y)
array([  7.27595761e-11,   2.91038304e-10,   6.54836184e-10, ...,
         1.18391205e+00,   1.18390998e+00,   1.18390790e+00])
>>> r(t).size, r(y).size
(951549, 951549)
The time difference (11.2 - 3.1 = 8.1 sec) is due to calculating spline built over these arrays onto "times" array

unestablished yet (http://openopt.org/unestablished):
r.extras is Python dict with fields startTimes, endTimes, infinums, supremums (arrays of same size to r(t).size), 
such that in interval (startTimes[i], endTimes[i])  infinums[i] <= y(t) <= supremums[i]
and supremums[i] - infinums[i] <= ftol
'''

# Now let's see a graphical visualization of splitted time intervals
# as you'll see, less time intervals are created where function is close to constanct
# and vise versa, more time intervals (with less size) are created
# near most problem regions, where function values change very intensively
# the peak near 4.321 (peak of our erf()) is near 750000 and doesn't fit into the resized picture
from pylab import hist, show, grid
hist(r(t), 5000)
grid('on')
show()
