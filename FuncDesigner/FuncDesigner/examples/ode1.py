""" Basic FuncDesigner example for solving ODE with automatic differentiation """

from FuncDesigner import *
from numpy import arange

# create some variables
a, b, c, t = oovars('a', 'b', 'c', 't')
# or just a, b, c, t = oovars(4)

# Python dict of ODEs: da/dt, db/dt, dc/dt
equations = {
             a: 2*a + cos(3*b-2*c) + exp(5-2*t), 
             b: arcsin(t/5) + 2*a + sinh(2**(-4*t)) + (2+t+sin(c))**(1e-1*(t-sin(a)+cos(b))), 
             c: a + 4*b - 45 - sin(100*t)
             }

startPoint = {a: 3, b: 4, c: 5}

timeArray = arange(0, 1, 0.01) # 0, 0.01, 0.02, 0.03, ..., 1

# assign ODE. 3rd argument (here "t") is time variable that is involved in differentiation.
myODE = ode(equations, startPoint, t, timeArray)

# Probably output will be changed till next release to more OpenOpt-like.
# Currently it's exact scipy.integrate.odeint output
r, infodict = myODE.solve()
print(r)
print(infodict['message'])
"""
...
 [ 279.24451223  218.05786991  336.11743436]
 [ 285.09427445  223.71515456  347.33272733]]
Integration successful.
"""
