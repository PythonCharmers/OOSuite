#f1 = integrate.scipy_quad(lambda x: sin(x) + cos(2*x), a, 2)
#f2 = integrate.scipy_quad(lambda x: sin(x) + cos(15*x), a, b)

# dirivateves will be obtained via automatic differentiation 
# http://en.wikipedia.org/wiki/Automatic_differentiation
# (not to be confused with symbolic differentiation
# and finite-differences derivatives approximation)

from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c') 
f1, f2 = sin(a) + cos(b) - log2(c) + sqrt(b), sum(c) + c * cosh(b) / arctan(a) + c[0] * c[1] + c[-1] / (a * c.size)
f3 = f1*f2 + 2*a + sin(b) * (1+2*c.size + 3*f2.size)
f = 2*a*b*c + f1*f2 + f3

point1 = {a:1, b:2, c:[3, 4, 5]}

print(f(point1))
print(f.D(point1))


#f3 = integrate.scipy_quad(lambda x: sin(x) + cos(2*x), F, 2*F)
"""
Expected output:
[ 11.69866225  15.69866225  19.69866225]
{'a': array([  8.83694535,  12.83694535,  16.83694535]), 'c': array([[ 4.,  0.,  0.],
       [ 0.,  4.,  0.],
       [ 0.,  0.,  4.]]), 'b': array([  7.2059025,   9.2059025,  11.2059025])}
"""
