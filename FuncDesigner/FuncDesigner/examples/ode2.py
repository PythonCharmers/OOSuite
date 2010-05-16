""" Advanced FuncDesigner example for solving ODE with automatic differentiation """

from FuncDesigner import *
from numpy import arange, zeros,  ones

N = 50

# create some variables
x, t = oovars('x', 't')
y = oovar('y', size = N)
z = oovar('z', size = 2*N)

# Python dict of ODEs: da/dt, db/dt, dc/dt
equations = {
             x: 2*x + exp(5-2*t), 
             y: arcsin(t/5) + 2*cos(y) + cos(sum(z)), # sum(z) and z.sum() are same, as well as for numpy arrays
             z: cos(z/10) + sin(x) + 4*cos(y.sum()) - 0.001*sinh(2*t) 
             }

startPoint = {x: 3, y: 4*ones(N), z: 5*zeros(2*N)}

timeArray = arange(0, 1, 0.01) # 0, 0.01, 0.02, 0.03, ..., 0.99

# assign ODE. 3rd argument (here "t") is time variable that is involved in differentiation.
myODE = ode(equations, startPoint, t, timeArray)

# Probably output will be changed till next release to more OpenOpt-like.
# Currently it's exact scipy.integrate.odeint output
r = myODE.solve()
X,  Y,  Z = r(x, y, z) # X.size = 100, Y.shape = (50, 100), Z.shape = (100, 100)
print(r.msg) # r.extras.infodict contains whole scipy.integrate.odeint infodict
