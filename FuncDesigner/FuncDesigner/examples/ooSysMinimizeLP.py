from FuncDesigner import *

a, b, c = oovars('a', 'b', 'c')
f = sum(a*[1, 2])+2*b+4*c
another_func = 2*f + 100 # doesn't matter in the example

startPoint = {a:[100, 12], b:2, c:40} # for to ajust sizes of the variables
S = oosystem(f, another_func)
# add some constraints
S &= [2*c+a-10 < 1500+0.1*b, a-10<150, c<300, a[0]>8.9, c>-100, 
      (a+b > [ 7.9, 7.8 ])('sum_a_b', tol=1.00000e-12), a > -10, b > -10, b<10]

r = S.minimize(f, startPoint) 
# you could use S.maximize as well

# default LP solvers are (sequentially, if installed): lpSolve, glpk, cvxopt_lp, lp:ipopt, lp:algencan, lp:scipy_slsqp, lp:ralg, lp:cobyla
# to change solver you can use kwarg "solver", e.g.
# r = S.minimize(f, startPoint, solver = 'cvxopt_lp')
# also you can provide any openopt kwargs:
# r = S.minimize(f, startPoint, iprint=-1, ...)

print(r.xf)
a_opt,  b_opt, c_opt = r(a, b, c)
# or any of the following: 
# a_opt,  b_opt, c_opt = r(a), r(b), r(c)
# r('a'), r('b'), r('c') (provided you have assigned the names to oovars as above)
# r('a', 'b', 'c')
# a(r), b(r), c(r)

"""
Expected output:
...
objFunValue: -375.5 (feasible, max(residuals/requiredTolerances) = 0)
{a: array([ 8.9, -2.2]), b: array([ 10.]), c: array([-100.])}
"""
