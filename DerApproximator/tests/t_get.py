from numpy import *
from DerApproximator import get_d1
func = lambda x: (x**4).sum()
x = arange(1.0, 6.0)

r1 = get_d1(func, x, stencil = 1, diffInt = 1e-4)
print(r1) # [   4.00060004   32.00240008  108.00540012  256.00960016  500.0150002 ]
r2 = get_d1(func, x, stencil = 2, diffInt = 1e-4)
print(r2) # [   4.00000004   32.00000008  108.00000012  256.00000016  500.0000002 ]
r3 = get_d1(func, x, stencil = 3, diffInt = 1e-4)
print(r3) # [   4.   32.  108.  256.  500.]
