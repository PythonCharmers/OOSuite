from numpy import *
from DerApproximator import *

func = lambda x: (x**4).sum()
func_d = lambda x: 40 * x**3 # incorrect; correct: 4 * x**3
x = arange(1.0, 6.0)
r = check_d1(func, func_d, x)
'''
 func num         user-supplied     numerical               RD
    0             +4.000e+01     +4.000e+00              3
    1             +3.200e+02     +3.200e+01              3
    2             +1.080e+03     +1.080e+02              3
    3             +2.560e+03     +2.560e+02              3
    4             +5.000e+03     +5.000e+02              3
max(abs(d_user - d_numerical)) = 4500.00002147
(is registered in func number 4)
***************************************************************************
'''

func = lambda x: x**4
func_d = lambda x: 40 * diag(x**3) # incorrect; correct: 4 * diag(x**3)
x = arange(1.0, 6.0)
r = check_d1(func, func_d, x)
'''
func num   i,j: dfunc[i]/dx[j]   user-supplied     numerical               RD
    0              0 / 0         +4.000e+01     +4.000e+00              3
    6              1 / 1         +3.200e+02     +3.200e+01              3
    12             2 / 2         +1.080e+03     +1.080e+02              3
    18             3 / 3         +2.560e+03     +2.560e+02              3
    24             4 / 4         +5.000e+03     +5.000e+02              3
max(abs(d_user - d_numerical)) = 4500.00002147
(is registered in func number 24)
***************************************************************************
'''
