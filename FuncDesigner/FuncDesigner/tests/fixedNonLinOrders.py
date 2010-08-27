from FuncDesigner import *

t=oovar('t')
f = sin(t)
assert f.getOrder(fixedVars=t) == 0
f2 = sin(f)
assert f2.getOrder(fixedVars=t) == 0
f3 = f + f2*f/(f2-f)
assert f3.getOrder(fixedVars=t) == 0

print 'OK'
