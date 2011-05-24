from FuncDesigner import *
from openopt import *

a, b, c, d, e, f = oovars('a', 'b', 'c', 'd', 'e', 'f')
F =  [
      (log(a+5)+cos(b) == 1.0)(tol=1e-10), 
      (a**3 + c == -1.5)(tol=1e-5), 
      d**3 + sqrt(abs(e)) == 0.5,#unassigned tol will be taken from p.ftol, default 10^-6
      abs(f)**1.5 + abs(b)**0.1 == 10,
      sinh(e)+arctan(c) == 4,  
      c**15 == d + e
      ]

startPoint = {a:0.51, b:0.52, c:0.53, d:0.54, e:0.55, f:0.56} # doesn't matter for interalg, matters for other solvers

solver='interalg' # set it to scipy_fsolve to ensure that the solver cannot handle the system
p = NLSP(F, startPoint, ftol = 1e-15)
# interalg requires finite box bounds on variables while scipy_fsolve cannot handle any constraints.
# To set box bounds you can do either 
#p.constraints = (a>-10, a<20, b>-20, b<10, c>-30, c<30, d>-32, d<32, e>-21, e<20, f>-10, f<10)
# or
p.implicitBounds=[-10, 10] # to affect all variables without assigned bounds

r = p.solve(solver, iprint = 0)
print(r(a, b, c, d, e, f))
'''
solver: interalg_0.21   problem: unnamed    type: NLSP
 iter   objFunVal   
    0  8.644e+00 
OpenOpt info: Solution with required tolerance 1.0e-15 
 is guarantied (obtained precision: 4.4e-16)
   91  4.441e-16 
istop: 1000 (optimal solution obtained)
Solver:   Time Elapsed = 0.68 	CPU Time Elapsed = 0.66
objFunValue: 4.4408921e-16 (feasible, MaxResidual = 0)
[-1.3563248494740674, -4.415033438436291, 0.9951183825864978, -0.9557107106921257, 1.88493645609756, 4.2752850759741]
solver: scipy_fsolve   problem: unnamed    type: NLSP
 iter   objFunVal   
    0  8.644e+00 
/usr/lib/python2.7/dist-packages/scipy/optimize/minpack.py:156: RuntimeWarning: The iteration is not making good progress, as measured by the 
  improvement from the last ten iterations.
  warnings.warn(msg, RuntimeWarning)
    1  8.644e+00 
istop: -101.0
Solver:   Time Elapsed = 0.04 	CPU Time Elapsed = 0.04
NO FEASIBLE SOLUTION is obtained (MaxResidual = 0, objFunc = 8.6442348)
[0.51, 0.52, 0.53, 0.54, 0.55, 0.56]
'''
