"""
Sparse SLE example
Under development yet
"""
from FuncDesigner import *
from time import time
t = time()
#def test():
n = 1000
# create some variables
a, b, c = oovar(), oovar(size=n), oovar(size=2*n)

f1 = a + sum(b*range(5, n+5))
f2 = a + 2*b + c.sum() 
f3 = a + a.size + 2*c.size 
f4 = c + range(4, 2*n+4) + 4*f3

f = [a+f4+5, 2*a+b*range(10, n+10)+15, a+4*b.sum()+2*c.sum()-45]
# alternatively, you could pass equations:
#f = [(a+f4).eq(-5), (2*a+b).eq(-15), a.eq(-4*b.sum()-2*c.sum()+45)]
linSys = sle(f)
r = linSys.solve()

A, B, C =  a(r), b(r), c(r)

print('A=%f B[4]=%f B[first]=%f C[last]=%f' % (A, B[4], B[0], C[-1]))

residuals = [F(r) for F in f]

print('time elapsed: %f' % (time()-t))
