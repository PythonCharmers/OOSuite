from numpy import *
from openopt import *

v = oovar('v', [-4, -5])
k = 50
f1 = oofun(lambda x: (x[0]+x[1]-15)**2 + k*(x[0]+2*x[1]-20)**2 - 400, input = v)
# exact min is (10, 5)

#f1.d=lambda x: [2*(x[0]+x[1]-15)+k*2*x[0]+k*4*x[1], 2*(x[0]+x[1]-20)+k*4*x[1]+k*4*x[0]]
f1.d = lambda x: [(x[0]-10)**2, (x[1]-5)**2]

f2 = oofun(lambda x: x[0]**2 + k*x[1]**2 - 300, d=lambda x:[2*x[0], 2*k*x[1]], input = v)

f3 = oofun(lambda x: (x[0]-100)**2+(x[1]-100)**2, input = v)
#f1.d = None
#f2.d = None
#f3 = oofun(lambda x, y: x+y, input = [f1, f2], d=lambda x, y:[1, 1])
p = NLP(f3, c = [f1, f2])
r=p.solve('scipy_cg')




