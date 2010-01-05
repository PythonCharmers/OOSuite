"""
FuncDesigner sparse SLE example
"""

from FuncDesigner import *
from time import time
t = time()

n = 1000

a, b, c = oovar(), oovar(size=n), oovar(size=2*n)

f1 = a + sum(b*range(5, n+5))
f2 = a + 2*b + c.sum() 
f3 = a + a.size + 2*c.size 
f4 = c + range(4, 2*n+4) + 4*f3

f = [a+f4+5, 2*a+b*range(10, n+10)+15, a+4*b.sum()+2*c.sum()-45]
# alternatively, you could pass equations:
#f = [(a+f4).eq(-5), (2*a+b).eq(-15), a.eq(-4*b.sum()-2*c.sum()+45)]
linSys = sle(f)

r = linSys.solve() # i.e. using autoselect - solver numpy.linalg.solve for dense (for current numpy 1.4 it's LAPACK dgesv)
# and scipy.sparse.linalg.spsolve for sparse SLEs (for current scipy 0.8 it's LAPACK dgessv)

A, B, C =  a(r), b(r), c(r) # or A, B, C = r[a], r[b], r[c]

print('A=%f B[4]=%f B[first]=%f C[last]=%f' % (A, B[4], B[0], C[-1]))
maxResidual = r.ff

# Note - time may differ due to different matrices obtained from SLE rendering
# because Python 2.6 doesn't has ordered sets (they are present in Python 3.x)
# maybe I'll implement fixed rendering in future for 3.x, I don't want to deal 
# with quite difficult walkaround for 2.6 
print('time elapsed: %f' % (time()-t))

