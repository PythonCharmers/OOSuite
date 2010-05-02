"""
Solving system of equations:
x**3 + y**3 - 9 = 0
x - 0.5*y = 0
cos(z) + x - 1.5 = 0
"""

from FuncDesigner import *
from openopt import NLSP

x, y, z = oovars(3)

f = (x**3 + y**3 - 9, x - 0.5*y, cos(z) + x - 1.5)

startPoint = {x:8, y:15, z:80}

p = NLSP(f, startPoint)

# optional: we could set some constraints
p.constraints = [z<70,  z>50,  z + sin(x) < 60]

r = p.solve('nssolve') # nssolve is name of solver involved, see OOF doc for more arguments
xs, ys, zs = r(x, y, z)
print('Solution: x = %f   y = %f  z = %f' % (xs, ys, zs))

