from FuncDesigner import *
from numpy import *

a, b = oovars('a','b')

point = {a:10, b:[3, 4, 5]}

f1 = [1, 2, 3] + 2*a + 4 + array((1, 2, 10))
f2 = [1, 2, 3] + 3*b + 4 + array((1, 2, 10))
f3 = [1, 2, 3]*a + 3*b*a + 4 + array((1, 2, 10))

for f in (f1,  f2, f3):
    print f(point)
    print f.D(point)
"""
[ 26.  28.  37.]
{'a': array([ 2.,  2.,  2.])}
[ 15.  20.  32.]
{'b': array([[ 3.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0.,  0.,  3.]])}
[ 105.  146.  194.]
{'a': array([ 10.,  14.,  18.]), 'b': array([[ 30.,   0.,   0.],
       [  0.,  30.,   0.],
       [  0.,   0.,  30.]])}
"""
