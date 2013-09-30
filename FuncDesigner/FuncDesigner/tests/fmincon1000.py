from FuncDesigner import *
from openopt import NLP
from numpy import arange

a, b, c = oovars('a b c')
c.ub = 1.06

n = 1000 

f = sum((a - arange(1, n+1) / 1000.0)**2) \
# that is (a_0-0.001)**2 + (a_1-0.002)**2 + ...  + (a_999-1)**2
+ (b-1.5)**2 + (c-15)**2

startPoint = {a:[0]*n, # vector with n zeros, however, you'd better use numpy arrays instead of Python lists
              b: 2, 
              c: 40} 
constraints = [(b-15)**2<4, c**4<0.0016, a[0] + a[1] > 2, b + sum(a) < 15]
p = NLP(f, startPoint, constraints=constraints)

r = p.solve('fmincon', matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# notebook Intel Atom 1.6 GHz:
#Solver:   Time Elapsed = 199.73	(includes MATLAB session create time ~ 30 sec)
#objFunValue: 252.99325 (feasible, MaxResidual = 7.82818e-10)
a_opt, b_opt, c_opt = r(a,b,c)
print(b_opt,c_opt)# (13.0, 0.20000002446305162)
print(a_opt[:5]) # [ 0.9995      1.0005     -0.49849998 -0.49749998 -0.49649998]
