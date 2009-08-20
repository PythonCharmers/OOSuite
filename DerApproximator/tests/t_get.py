from numpy import *
from DerApproximator import get_d1
func = lambda x: (x**4).sum()
x = arange(1.0, 6.0)

r1 = get_d1(func, x, stencil = 1, diffInt = 1e-5)
print(r1)
r2 = get_d1(func, x, stencil = 2, diffInt = 1e-5)
print(r2)

