from FuncDesigner import *
from openopt import MINLP

a, b, c = oovars('a', 'b', 'c')
d = oovar('d', domain = [1, 2, 3, 3.5, -0.5,  -4]) # domain should be Python list/set/tuple of allowed values
startPoint = {a:[100, 12], b:2, c:40, d:0} # however, you'd better use numpy arrays instead of Python lists

f = sum(a*[1, 2])**2 + b**2 + c**2 + d**2
constraints = [(2*c+a-10)**2 < 1.5 + 0.1*b, (a-10)**2<1.5, a[0]>8.9, (a+b > [ 7.97999836, 7.8552538 ])('sum_a_b', tol=1.00000e-12), \
a < 9, b < -1.02, c > 1.01, ((b + c * log10(a).sum() - 1) ** 2==0)(tol=1e-6), b+d**2 < 100]

p = MINLP(f, startPoint, constraints = constraints)
r = p.minimize('branb', nlpSolver='ralg', plot=0, discrtol = 1e-6, xtol=1e-7)
#r = p.solve('ralg') # for NLPs old-style (openopt 0.25 and below) p.solve() is same to p.minimize()
#r = p.maximize('ralg')
print(r.xf)
a_opt,  b_opt, c_opt, d_opt = r(a, b, c, d)
# or any of the following: 
# a_opt,  b_opt, c_opt,d_opt = r(a), r(b), r(c),r(d)
# r('a'), r('b'), r('c'), r('d') (provided you have assigned the names to oovars as above)
# r('a', 'b', 'c', 'd')
# a(r), b(r), c(r), d(r)

"""
Expected output:
...
objFunValue: 718.00734 (feasible, max(residuals/requiredTolerances) = 0.135819)
{a: array([ 8.99999879,  8.87525369]), b: array([-1.01999986]), c: array([ 1.06177278]), d: array([-0.5])}
"""
