"""
Another one, more advanced example
for solving SLE (system of linear equations)
"""
from FuncDesigner import *
n = 100
# create some variables
a, b, c = oovar(), oovar(size=n), oovar(size=2*n)

# let's construct some linear functions R^i -> R^j
# in Python range(m, k) is [m, m+1, ..., m+k-1]
f1 = a + sum(b*range(5, n+5)) # R^(n+1) -> R
f2 = a + 2*b + c.sum() # R^(2n+1) -> R^(n)

# you could use size of oovars
f3 = a + a.size + 2*c.size # R^(2n+1) -> R; a.size and c.size will be resolved into 1 and 2*n

f4 = c + range(4, 2*n+4) + f1 + 0.5*f2.sum() + 4*f3

# We can use "for" cycle:
for i in xrange(4):
    f4 = 0.5*f4 + a + f1 + 1

# Python list of linear equations 
f = [a+f4+5, 2*a+b*range(10, n+10)+15, a+4*b.sum()+2*c.sum()-45]
# alternatively, you could pass equations:
#f = [(a+f4).eq(-5), (2*a+b).eq(-15), a.eq(-4*b.sum()-2*c.sum()+45)]

linSys = sle(f)
r = linSys.solve()

# get result
A, B, C =  a(r), b(r), c(r)
# or 
# A, B, C =  r[a], r[b], r[c]
