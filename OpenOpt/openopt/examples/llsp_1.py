__docformat__ = "restructuredtext en"

from numpy import empty, sin, cos, arange
from openopt import LLSP

M, N = 3, 2
C = empty((M,N))
d =  empty(M)

for j in xrange(M):
    d[j] = 1.5*N+80*sin(j)
    C[j] = 8*sin(4.0+arange(N)) + 15*cos(j)

p = LLSP(C, d)
#r = p.solve('lapack_dgelss') #requires scipy installed
#p.debug=1
#r = p.solve('toms587')
r = p.solve('cvxopt_llsp')
# or using single-precision:
#r = p.solve('lapack_sgelss')

#using llsp2nlp converter with an NLP solver:
#r = p.solve('nlp:scipy_cg')

print 'f_opt:', r.ff # 1601171.51977
#print 'x_opt:', r.xf

