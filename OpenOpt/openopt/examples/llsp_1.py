__docformat__ = "restructuredtext en"

from numpy import empty, sin, cos, arange
from openopt import LLSP

M, N = 1500, 1000
C = empty((M,N))
d =  empty(M)

for j in xrange(M):
    d[j] = 1.5*N+80*sin(j)
    C[j] = 8*sin(4.0+arange(N)) + 15*cos(j)
    
""" alternatively, try the sparse problem - lsqr solver can take benefits of it.
Also, if your C is too large for your RAM 
you can pass C of any scipy.sparse matrix format

for j in xrange(M):
    d[j] = 1.5*N+80*sin(j)
    C[j, j%N] = 15*cos(j) #+ 8*sin(4.0+arange(N))
    C[j, (1 + j)%N] = 15*cos(j) #+ 8*sin(4.0+arange(N))
"""

p = LLSP(C, d)
r = p.solve('lsqr')

print 'f_opt:', r.ff # 2398301.68347
#print 'x_opt:', r.xf

