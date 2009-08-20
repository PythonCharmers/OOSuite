"""
OpenOpt SOCP example
This one is same to the sample from
http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#second-order-cone-programming
"""

from numpy import *
from openopt import SOCP

f = array([-2, 1, 5])

C0 = mat('-13 3 5; -12 12 -6')
d0 = [-3, -2]
q0 = array([-12, -6, 5])
s0 = -12

C1 = mat('-3 6 2; 1 9 2; -1 -19 3')
d1 = [0, 3, -42]
q1 = array([-3, 6, -10])
s1 = 27

p = SOCP(f,  C=[C0, C1],  d=[d0, d1], q=[q0, q1], s=[s0, s1])
r = p.solve('cvxopt_socp')
x_opt, f_opt = r.xf,  r.ff
print ' f_opt:', f_opt, '\n x_opt:', x_opt
# f_opt: -38.3463678559 
# x_opt: [-5.01428121 -5.76680444 -8.52162517]
