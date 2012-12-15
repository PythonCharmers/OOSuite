""" Basic FuncDesigner example for solving ODE with automatic differentiation """

from FuncDesigner import *
from numpy import arange

# create some variables
x, y, z, t = oovars('x', 'y', 'z', 't')
# or just x, y, z, t = oovars(4)

# Python dict of ODEs
equations = {
             x: 2*x + cos(3*y-2*z) + exp(5-2*t), # dx/dt
             y: arcsin(t/5) + 2*x + sinh(2**(-4*t)) + (2+t+sin(z))**(1e-1*(t-sin(x)+cos(y))), # dy/dt
             z: x + 4*y - 45 - sin(100*t) # dz/dt
             }

startPoint = {x: 3, y: 4, z: 5}

timeArray = arange(0, 1, 0.01) # 0, 0.01, 0.02, 0.03, ..., 0.99

# assign ODE. 3rd argument (here "t") is time variable that is involved in differentiation.
myODE = ode(equations, startPoint, {t: timeArray})

r = myODE.solve()
X,  Y,  Z = r(x, y, z)
print(X[50:55], Y[50:55], Z[50:55])
print(r.msg) # r.extras.infodict contains whole scipy.integrate.odeint infodict
"""
(array([  95.32215541,   97.80251715,  100.32319065,  102.88116657, 105.47545552]), 
array([ 50.26075725,  52.20594513,  54.2008745 ,  56.24637088,  58.34441539]), 
array([ 44.40064889,  46.96317981,  49.62269611,  52.38990533,  55.2741921 ]))
Integration successful.
"""
