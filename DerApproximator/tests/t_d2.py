from DerApproximator import *
from numpy import *
ff=lambda x: sum(x**2) + x[0]*x[2]

x = [1, 2, 3]

print(get_d2(ff, x))
