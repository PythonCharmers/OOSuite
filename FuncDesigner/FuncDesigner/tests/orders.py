from numpy import inf
from FuncDesigner import *
a, b=oovars('a', 'b')

# test +
c1 = a+b
c2 = a+4

assert c1.getOrder(fixedVars=a) == 1 and c1.getOrder(fixedVars=[a, b]) == 0
assert (a.getOrder(), a.getOrder(a), a.getOrder(fixedVars=a)) == (1, 1, 0)
assert (c1.getOrder(), c2.getOrder()) == (1, 1)
assert (c1.getOrder(a), c2.getOrder(a)) == (1, 1)
assert (c2.getOrder(fixedVars=a), c1.getOrder(fixedVars=[a, b])) == (0, 0)

# test -
c3 = -a; assert c3.getOrder() == 1 and c3.getOrder(fixedVars=a) == 0 
c4 = a-b
assert c4.getOrder() == 1 and c4.getOrder(fixedVars=a) == 1 and c4.getOrder(fixedVars=b) == 1
assert c4.getOrder(fixedVars=[a, b]) == 0

# test *
c5 = [1, 2]*2*(2*a); assert c5.getOrder() == 1 and c5.getOrder(fixedVars=a) == 0
c6 = a*b; assert c6.getOrder() == 2 and c6.getOrder(fixedVars=a) == 1

c7 = c6*a
assert c7.getOrder() == 3
assert c7.getOrder(fixedVars=[a, b]) == 0 and c6.getOrder(fixedVars=[a, b]) == 0

c8 = c7*c7; assert c8.getOrder() == 6

# test /
c9 = a/[1, 2]; assert c9.getOrder() == 1 and c9.getOrder(fixedVars=a) == 0
c10 = [1, 2]/a; assert c10.getOrder() == inf and c10.getOrder(fixedVars=a) == 0
c11 = b/a; assert c11.getOrder() == inf and c11.getOrder(fixedVars=a) == 1 and c11.getOrder(fixedVars=b) == inf and c11.getOrder(fixedVars=[a, b]) == 0

# test ^
c12 = a**b
assert c12.getOrder() == inf

# test sum(oofuns)
c13 = sum([a, b]); assert c13.getOrder() == 1 and c13.getOrder(fixedVars=a) == 1 and c13.getOrder(fixedVars=[a, b]) == 0

# test a[ind]
c14 = (a*b)[0]; assert c14.getOrder() == 2 and c14.getOrder(fixedVars=a) == 1

# test a[ind1:ind2]
c15 = (a*b)[0:15]; assert c15.getOrder() == 2 and c15.getOrder(fixedVars=a) == 1

# test oofun.sum()
c16 = (a*b).sum(); assert c16.getOrder() == 2 and c16.getOrder(fixedVars=a) == 1

# test >
c17 = a>b; assert c17.getOrder() == inf

# test sin
c18 = sin(a); assert c18.getOrder() == inf

# test ifThenElse
c19 = ifThenElse(a>1, a, b); assert c19.getOrder() == inf
# TODO: try to set correct value from val1, val2 if condition is fixed

print 'passed'
