from FuncDesigner import *
N = 100
a = oovars(N) # create array of N oovars
b = oovars(N) # another array of N oovars 
some_lin_funcs = [i*a[i]+4*i + 5*b[i] for i in xrange(N)]
f = some_lin_funcs[15] + some_lin_funcs[80] - sum(a) + sum(b)
point = {}
for i in xrange(N):
    point[a[i]] = 1.5 * i**2
    point[b[i]] = 1.5 * i**3
print f(point) # prints 40899980.

from openopt import LP
p = LP(f, point)
aLBs = [a[i]>-10 for i in xrange(N)]
bLBs = [b[i]>-10 for i in xrange(N)]
aUBs = [a[i]<15 for i in xrange(N)]
bUBs = [b[i]<15 for i in xrange(N)]
p.constraints = aLBs + bLBs + aUBs + bUBs
p.constraints.append(a[4]+b[15]>-9)
# or p.constraints += [a[4]+b[15]>-9]
r = p.solve('glpk')
print('opt a[15]=%f'%r.xf[a[15]]) 
# opt a[15]=-10.000000
