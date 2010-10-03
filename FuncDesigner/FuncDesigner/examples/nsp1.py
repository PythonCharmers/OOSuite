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
x_opt, y_opt = r(x, y)
print(max(abs(x_opt)), y_opt)
"""
...
  990  3.000e+01 
istop: 4(|| F[k] - F[k-1] || < ftol)
Solver:   Time Elapsed = 11.3 	CPU Time Elapsed = 10.88
objFunValue: 30.000409
(2.3481047305075244e-06, array([  6.18404460e-08]))
"""
