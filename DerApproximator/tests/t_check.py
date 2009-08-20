from numpy import *
from DerApproximator import *

func = lambda x: (x**4).sum()
func_d = lambda x: 40 * x**3
x = arange(1.0, 6.0)

r = check_d1(func, func_d, x)

######################################
func = lambda x: x**4
func_d = lambda x: 40 * diag(x**3)
x = arange(1.0, 6.0)

r = check_d1(func, func_d, x)


