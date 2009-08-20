from numpy import *
from openopt.kernel.fdda import get_d1
func = lambda x: (x**4).sum()
x = arange(1.0, 6.0)

r = get_d1(func, x, twoSized = False, diffInt = 1e-4)
print r

