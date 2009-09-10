"""
GLP (GLobal Problem from OpenOpt set) example for FuncDesigner model:
searching for global minimum of the func 
(x-1.5)**2 + sin(0.8 * y ** 2 + 15)**4 + cos(0.8 * z ** 2 + 15)**4 + (t-7.5)**4
subjected to -1<= var <= 1 for any var from [x,y,z,t] 
See http://openopt.org/GLP for more info and examples.
"""
from openopt import GLP
from FuncDesigner import *

x, y, z, t = oovars(4)

f = (x-1.5)**2 + sin(0.8 * y ** 2 + 15)**4 + cos(0.8 * z ** 2 + 15)**4 + (t-7.5)**4
constraints = [x<1, x>-1, y<1, y>-1, z<1, z>-1, t<1, t>-1]
startPoint = {x:0, y:0, z:0, t:0}
p = GLP(f, startPoint, constraints=constraints,  maxIter = 1e3,  maxFunEvals = 1e5,  maxTime = 3,  maxCPUTime = 3)

#optional: graphic output
#p.plot = 1 or p.solve(..., plot=1) or p = GLP(..., plot=1)

r = p.solve('de', plot=1) # try other solvers: galileo, pswarm
x_opt,  f_opt = r.xf,  r.ff
