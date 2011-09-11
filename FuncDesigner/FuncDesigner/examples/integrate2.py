from numpy import pi
import sys
sys.path += ['/home/dmitrey/OOSuite/FuncDesigner', '/home/dmitrey/OOSuite/SpaceFuncs', '/home/dmitrey/OOSuite/OpenOpt']

sigma = 1e-4
ff = lambda y, x:  (exp(-(x-0.1)**2/(2*sigma)) * exp(-(y+0.2)**2/(2*sigma))) / (2*pi*sigma)
#ff = lambda y, x: 100*cos(y+x)*sin(y+x**2)

bounds_x = (-15, 5)
bounds_y = (-15, 5)

from FuncDesigner import *
from openopt import IP
x, y = oovars('x y') 
#f = ff(y, x) 
# or 
f = (exp(-(x-0.1)**2/(2*sigma)) * exp(-(y+0.2)**2/(2*sigma))) / (2*pi*sigma)
#f = 100*cos(y+x)*sin(y+x**2)

domain = {x: bounds_x, y: bounds_y}
p = IP(f, domain, ftol = 5e-2)
r = p.solve('interalg', maxIter = 15000, maxNodes = 500000, maxActiveNodes = 150, iprint = 100)
print('interalg result: %f' % p._F)
'''
Solver:   Time Elapsed = 0.72 	CPU Time Elapsed = 0.7
interalg result: 1.007690 (usually solution, obtained by interalg, has real residual 10-100 times less 
than required tolerance, because interalg works with "most worst case" that extremely rarely occurs. 
Unfortunately, real obtained residual cannot be revealed).
Now let's ensure scipy.integrate dblquad fails to solve the problem and mere lies about obtained residual:'''

from scipy.integrate import dblquad
val, abserr = dblquad(ff, bounds_x[0], bounds_x[1], lambda y: bounds_y[0], lambda y: bounds_y[1])

print('scipy.integrate dblquad value: %f   declared residual: %f' % (val, abserr)) 
''' scipy.integrate dblquad value: 0.000000   declared residual: 0.000000
While scipy dblquad fails already for sigma = 10^-4, interalg works perfectly even for sigma  = 10^-14:
Solver:   Time Elapsed = 121.15 	CPU Time Elapsed = 119.89
interalg result: 1.003149 '''
