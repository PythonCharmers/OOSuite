from FuncDesigner import *
from openopt import *

a, b, c, d, e, f = oovars('a', 'b', 'c', 'd', 'e', 'f')

# objective function:
F = cos(5*a) + (a-0.9)**2 +exp(10*abs(a-0.1)) \
+ cos(10*b) + (b-0.2)**2 + exp(10*abs(b-0.7)) \
+ cos(20*c) + (c-0.3)**2  + exp(10*abs(c-0.9)) \
+ cos(40*d) + (d-0.4)**2 + exp(e/4) + cos(5*f)

startPoint = {a:0.5, b:0.5, c:0.5, d:0.5, e:0.5, f:0.5}

# set box-bound domain:
constraints = (a>0, a<1, b>0, b<1, c>0, c<1, d>0, d<1, e>0, e<1, f>0, f<1)

# choose required objective function tolerance: 
# |f-f*| < fTol, where f* is objective function value in optimal point
fTol = 5e-4

solver='interalg'
# other global solvers to compare (they cannot handle required tolerance fTol)
#solver=oosolver('mlsl', iprint = 200, maxIter = 10000)
#solver=oosolver('de', iprint=10, maxFunEvals = 10000, maxIter = 1500)
# or this solver with some non-default parameters:
#solver=oosolver('interalg', fStart = 5.175, maxIter = 1000,maxNodes = 1000000, maxActiveNodes = 15)

p = GLP(F, startPoint, fTol = fTol, constraints = constraints)
r = p.minimize(solver)
print(r.xf)#{a: 0.100006103515625, b: 0.70000457763671875, c: 0.83860015869140625, d: 0.392822265625, e: 0.0001220703125, f: 0.62890625}

'''                                                    Some other objectives that were tried with interalg                                                    '''
#F = cos(5*a) + (a-0.1)**2 + cos(10*b) + (b-0.2)**2 + cos(20*c) + (c-0.3)**2  + cos(40*d) + (d-0.4)**2 + exp(e/4) + cos(5*f)
#F = 5*sin(5*a) +ceil(sin(10*a) + exp(5*a))+ (a-0.1)**2+ exp(0.1*(b + c)) + 5*cos(5*d) * cos(5*a) + exp(d/5) + cos(e) + cos(f) 
#F = sin(0) * a ** 0 + sin(1) * a ** 1
#F = (a-0.1)*a*b*c + a*(b-0.2)*b + (c-0.3)*c * f*d*e+ (f+0.2)**3*a*b*(c-0.5) + d*e*f + a/((f-0.95)**2)
#F =  (a-0.1)*a*b*c + a*(b-0.2)*b + (c-0.3)*c * f*d*e+ (f+0.2)**3*a*b*(c-0.5) + d*e*f + a/((f-0.95)**2)
#F =  a/((f-0.95)**2)
#F = (a+1e-10)/(f-0.95)**2
#F = a**3 + (a-0.1)**2 + b**3 + (b-0.2)**2 + c**5 + (c-0.3)**2  + d**3 + (d-0.4)**2 + exp(e/4) + a*b*(f-0.7)*3
#F = exp(a) + exp(b)
#F = a+b+exp(c)/1e100

#N = 5
#a = oovars(N)
#F = sum([sin(1+N*i)*a[i]**int(2+i**0.3) for i in range(N)])
#S = oosystem()
#S &= [a[i] > 0 for i in range(N)] + [a[i] < 1 for i in range(N)]
#startPoint = dict([(a[i], 0.5) for i in range(N)])
#startPoint = {a:[1]*N}
#r = S.minimize(F, startPoint, solver=solver)
#print(r.xf)
