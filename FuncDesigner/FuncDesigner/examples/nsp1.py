"""
Example for NSP 
|x[0]| + 1.2*|x[1]| + 1.44*|x[2]| + ... + 1.2^74*|x[74]| + abs(y-15) +abs(y+15) + y^2 +  -> min
"""
from FuncDesigner import *
from openopt import NSP
from numpy import cos, arange

x, y = oovars('x', 'y')

N = 75
koeffs = arange(1, N+1) ** 1.2

f = (abs(x) * koeffs).sum() + abs(y-15) + abs(y+15) + y**2

startPoint = {x: cos(1+arange(N)), y:80}

p = NSP(f, startPoint, maxIter = 1e5)

r = p.solve('ralg')
x_opt, y_opt = r.xf[x], r.xf[y]
print(max(abs(x_opt)), y_opt)
"""
...
 1129  3.000e+01 
istop:  4 (|| F[k] - F[k-1] || < ftol)
Solver:   Time Elapsed = 9.27 	CPU Time Elapsed = 8.62
objFunValue: 30.002138
(9.6909792750048539e-06, array([  1.03949811e-05]))
"""
