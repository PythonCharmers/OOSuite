from numpy import pi
sigma = 1e-5
# here order z, y, x is used instead of x,y,z  to keep it in accordance with scipy.integrate.tplquad API
# for FuncDesigner models you shouldn't keep the order in mind
ff = lambda z, y, x:  exp(-(x+0.15)**2/(2*sigma)) / sqrt(2*pi*sigma) 
'''Pay attention: 1st part is positive,
for 2nd part sin(z) = -sin(-z) and thus integraion over z = (-val, val)
must yield zero, hence result has to be positive, 
but scipy tplquad says it's zero'''

bounds_x = (-5, 4)
bounds_y = (-0.5, 0.1)
bounds_z = (-0.2, 0.2)
# expected result: (1-epsilon) * (0.1 - (-0.5)) * (0.2 - (-0.2)) ~= 1 * 0.6 * 0.4 = 0.24

from FuncDesigner import *
from openopt import IP
x, y, z = oovars('x y z') 
# f = ff(z, y, x)
# or 
f = exp(-(x+0.15)**2/(2*sigma)) / sqrt(2*pi*sigma) + cos(y)*sin(z)*cos(2*x) + 1e-1*x*y*z

domain = {x: bounds_x, y: bounds_y,  z: bounds_z}
p = IP(f, domain, ftol = 0.05)
r = p.solve('interalg', maxIter = 50000, maxActiveNodes = 150, maxNodes = 500000, iprint = 100)
print('interalg result: %f' % p._F)
''' Solver:   Time Elapsed = 2.03 	CPU Time Elapsed = 2.0
objFunValue: 0.24019923 (feasible, MaxResidual = 0.0470517)
interalg result: 0.240199 (usually solution, obtained by interalg, has real residual 10-100 times less 
than required tolerance, because interalg works with "most worst case" that extremely rarely occurs. 
Unfortunately, real obtained residual cannot be revealed).
Now let's ensure scipy.integrate tplquad fails to solve the problem and mere lies about obtained residual:'''

from scipy.integrate import tplquad
val, abserr = tplquad(ff, bounds_x[0], bounds_x[1], lambda y: bounds_y[0], lambda y: bounds_y[1], \
                      lambda y, z: bounds_z[0], lambda y, z: bounds_z[1])
print('scipy.integrate dblquad value: %f   declared residual: %f' % (val, abserr)) 
''' scipy.integrate tplquad value: 0.000000   declared residual: 0.000000
While scipy tplquad fails already for sigma = 10^-5, interalg works perfectly even for sigma  = 10^-10:
Solver:   Time Elapsed = 2.05 	CPU Time Elapsed = 2.06
interalg result: 0.240005 '''

# for bounds_z = (-0.2, 0.34) and sigma = 10^-5 interalg result: 0.328808 (ftol = 0.05, time elapsed ~ 1 min)
# scipy.integrate dblquad value: 0.004813   declared residual: 0.000000
