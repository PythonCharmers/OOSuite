from FuncDesigner import *
from openopt import NLP, oosolver
from numpy import arange
N = 10
a = oovar('a', size=N)
b = oovar('b', size=N)
f = (sum(abs(a*arange(1,N+1))**2.5)+sum(abs(b*arange(1,N+1))**1.5)) / 10000

startPoint = {a:cos(arange(N)), b:sin(arange(N))} # however, you'd better use numpy arrays instead of Python lists
p = NLP(f, startPoint)
#p.constraints = [(a>1)(tol=1e-5), a**2+b**2 < 0.1+a+b]# + [(a[i]**2+b[i]**2 < 3+abs(a[i])+abs(b[i])) for i in range(N)]

p.constraints = [a>1, 10*a>10]          # + [(a[i]**2+b[i]**2 < 3+abs(a[i])+abs(b[i])) for i in range(N)]
#p.constraints = []
#C = sum(a**2+b**2 - (0.1+a+b)
C = [ifThenElse(a[i]**2+b[i]**2 - (0.1+a[i]+b[i])<0,  0,  a[i]**2+b[i]**2 - (0.1+a[i]+b[i])) for i in range(N)]
p.constraints.append(sum(C)<0)
solver = 'ipopt'
solver = oosolver('ralg')
#solver = oosolver('ralg', approach='nqp')
r = p.minimize(solver, maxIter = 15000)
