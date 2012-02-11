'''
An example of solving global optimization problem
with guaranteed precision |f-f*| < fTol
'''
from numpy import zeros
from FuncDesigner import *
from openopt import *

a, b, c = oovars(3) # create 3 variables
d = oovars(4) # create 4 variables in a single vector

# objective function:
F = cos(5*a) + 0.2*(b-0.2)**2 +  exp(4*abs(c-0.9)) \
+ 0.05*sum(sin(d-0.1*(a+b+c))) + 3 * abs(d[0] - 0.2) + 4 * abs(d[2] - 0.2)

startPoint = {a:0.5, b:0.5, c:0.5, d: zeros(4)}

# set box-bound domain:
constraints = [a>0, a<1, b>0, b<1, c>0, c<1, d>-1, d<1, d[3] < 0.5]

# set some general constraints:
constraints += [
                (a*b + sin(c) < 0.5)(tol=1e-5), 
                d < cos(a) + 0.5, # default tol 10^-6
                cos(d[0]) +a < sin(d[3]) + b, 
                d[1] + c == 0.7
                ]

# choose required objective function tolerance: 
# |f-f*| < fTol, where f* is objective function value in optimal point
fTol = 0.005

solver='interalg'
# another global solver to compare (it cannot handle required tolerance fTol)
#solver=oosolver('de', iprint=10, maxFunEvals = 10000, maxIter = 1500)
# or this solver with some non-default parameters:
#solver=oosolver('interalg', fStart = 5.175, maxIter = 1000,maxNodes = 1000000, maxActiveNodes = 15)

p = GLP(F, startPoint, fTol = fTol, constraints = constraints)
r = p.minimize(solver, iprint = 200)
print(r(a, b, c, d))
'''
------------------------- OpenOpt 0.37 -------------------------
solver: interalg   problem: unnamed    type: GLP
 iter   objFunVal   log10(MaxResidual/ConTol)   
    0  5.540e+00                      6.00 
  200  5.975e+00                     -0.77 
  400  5.602e+00                     -0.87 
OpenOpt info: Solution with required tolerance 5.0e-03 
 is guarantied (obtained precision: 4.9e-03)
  477  5.561e+00                     -0.11 
istop: 1000 (solution has been obtained)
Solver:   Time Elapsed = 41.26 	CPU Time Elapsed = 40.38
objFunValue: 5.5611794 (feasible, max(residuals/requiredTolerances) = 0.783904)
[6.4849853515625e-05, 0.52830866626324779, 0.52356340597460604, 
[0.20009332597693788, 0.1764373779296875, 0.19999030162612966, 0.48973883274923724]]
'''

