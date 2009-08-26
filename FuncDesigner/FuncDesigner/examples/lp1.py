# Example of creating LP in FuncDesigner
# and solving it via OpenOpt

from FuncDesigner import *
from openopt import LP

# Define some oovars
x, y, z = oovars(3)

# Let's define some linear functions
f1 = 4*x+5*y + 3*z + 5
f2 = f1.sum() + 2*x + 4*y + 15
f3 = 5*f1 + 4*f2 + 20

# Define objective
obj = x.sum() + y - 50*z + sum(f3) + 2*f2.sum() + 4064.6

# Start point - currently matters only size of variables
startPoint = {x:[8, 15], y:25, z:80} # however, using numpy.arrays is more recommended than Python lists

# Create prob
p = LP(obj, startPoint)

# Define some constraints
p.constraints = [x+5*y<15, x[0]<4, f1<[25, 35], f1>-100, 2*f1+4*z<[80, 800], 5*f2+4*z<100, -5<x,  x<1, -20<y,  y<20, -4000<z, z<4]

# Solve
r = p.solve('glpk') # glpk is name of solver involved, see OOF doc for more arguments

# Decode solution
s = r.xf
print('Solution: x = %s   y = %f  z = %f' % (str(s[x]), s[y], s[z]))
# Solution: x = [-5. -5.]   y = -19.400000  z = 4.000000
