'''
interalg example for global MINLP with 9 variables and some constraints
'''
from FuncDesigner import *
from openopt import *
 
n = 7
x = oovars(n)
x[4].domain = [0.9, -0.7, 0.4] # let's set one of x coords as  discrete variable
y = oovar(domain = bool) # the same to domain = [0,1]
z = oovar(domain = {1, 1.5, 2, 4, 6, -4, 5}) # for obsolete Python versions use [] or () instead of {}

F = abs(x[0]-0.1) + abs(x[n-1]-0.9)  +0.55 * (x[0]-0.1) * (0.2-x[3]) +  1e-1*sum(x)  + y*sin(z) 

constraints = [
               x>-1, x<1, 
               (x[0]**2 + x[1]**2 == 0.5)(tol=1.0e-7), 
               (x[1]**2 + 0.71*x[1]**2 + (x[2]-0.02)**2 <= 0.17)(tol=1e-4), 
               (x[2]-0.1)**2 + (x[3]-0.03)**2 <= 0.01,  # default constraint tol is 10^-6
               cos(y) + x[5] <= -0.1, 
               z**2 +arctan(x[1]) < 16, 
               interpolator([1, 2, 3, 4], [1.001, 4, 9, 16.01])(x[4]+2*x[5]) < 6
               ]

startPoint = {x:[0]*n, y:0, z:1.5} # [0]*n means Python list [0,0,...,0] with n zeros

# interalg solves problems with specifiable accuracy fTol: 
# | f - f*|< fTol , where f* is  theoretical optimal value

p = GLP(F, startPoint, fTol = 0.0005, constraints = constraints)
# interalg requires all finite box bounds, but they can be very huge, e.g. +/- 10^15
# you may found useful arg implicitBounds, for example p.implicitBounds = [-1, 1], 
# for those variables that haven't assigned bounds, 
# it affects only solvers that demand finite box bounds on variables

r = p.solve('interalg', iprint = 100)
print(r(x, y, z))
'''
------------------------- OpenOpt 0.37 -------------------------
solver: interalg   problem: unnamed    type: GLP
 iter   objFunVal   log10(MaxResidual/ConTol)   
    0  9.890e-01                      6.70 
  100  1.254e+00                     -1.64 
  200  1.221e+00                     -0.24 
OpenOpt info: Solution with required tolerance 5.0e-04 
 is guarantied (obtained precision: 5.0e-04)
  233  1.221e+00                     -0.01 
istop: 1000 (solution has been obtained)
Solver:   Time Elapsed = 16.86 	CPU Time Elapsed = 16.61
objFunValue: 1.2209237 (feasible, max(residuals/requiredTolerances) = 0.982429)
[[-0.63423135876655579, -0.31265074014663696, 0.073974609375, -0.0665283203125, -0.69999999999999996, -0.88623046875, 0.89990234375], 1.0, -4.0]
'''
