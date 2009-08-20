from FuncDesigner import *
from openopt import NLP
a, b, c = oovars('a', 'b', 'c')
f = sum(a*[1, 2])**2+b**2+c**2
startPoint = {a:[100, 12], b:2, c:40}
p = NLP(f, startPoint)
p.constraints = [(2*c+a-10)**2<1.5, (a-10)**2<1.5, a[0]>8.9, a+b>[ 7.97999836,  7.8552538 ], \
                 a<9, (c-2)**2<1, b<-1.02, c>1.01, log2(c-15*a/b)+a>4, ((b+2*c-1)**2).eq(0)]
r = p.solve('ralg', iprint =1)
print r.xf
# {a: array([ 8.99999769,  8.87525255]), b: array([-1.0199994]), c: array([ 1.01000874])}
