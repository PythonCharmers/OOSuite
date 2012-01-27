from DerApproximator import *
from numpy import *
ff=lambda x: sum(x**2) + x[0]*x[2]

x = [1, 2, 3]

print(get_d2(ff, x))
'''
[[  2.00000008e+00   7.01739901e-08   9.99999692e-01]
 [  7.01774441e-08   1.99999976e+00  -2.80707803e-07]
 [  9.99999692e-01  -2.80703855e-07   1.99999994e+00]]
'''
