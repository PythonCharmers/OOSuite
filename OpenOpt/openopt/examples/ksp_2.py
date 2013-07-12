#!/usr/bin/python
'''
Simplest OpenOpt KSP example;
requires FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/KSP for more details
'''
from openopt import *
from numpy import sin, cos

N = 1500

items = [
         {'name': 'item %d' % i,'cost': 1.5*(cos(i)+1)**2, 
         'volume': 2*sin(i) + 3, 
         'n':  1 if i < N/3 else 2 if i < 2*N/3 else 3} for i in range(N)
         ]
         
constraints = lambda values: (
                              values['volume'] < 10, 
                              values['nItems'] <= 10, 
                              values['nItems'] >= 5
                              )

objective = lambda val:  2*val['volume'] - val['cost'] + 3*val['nItems']
p = KSP(objective, items, goal = 'min', constraints = constraints) 
r = p.solve('glpk', iprint = 0) # requires cvxopt and glpk installed, see http://openopt.org/KSP for other solvers
''' Results for Intel Atom 1.6 GHz:
Initialization: Time = 7.0 CPUTime = 5.9
------------------------- OpenOpt 0.50 -------------------------
solver: glpk   problem: unnamed    type: MILP   goal: min
 iter   objFunVal   log10(maxResidual)   
    0  0.000e+00               0.70 
    1  8.699e+00            -100.00 
istop: 1000 (optimal)
Solver:   Time Elapsed = 5.39 	CPU Time Elapsed = 4.72
objFunValue: 8.6992982 (feasible, MaxResidual = 0
'''
print(r.xf) # {'item 546': 2, 'item 1256': 3}
# pay attention that Python indexation starts from zero: item 0, item 1 ...
# if fields 'name' are absent in items, you'll have list of numbers instead of Python dict
