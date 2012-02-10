'''
A MOP example for interalg 
'''
from FuncDesigner import *
from openopt import *



S = oovar('cooler area')
V = oovar('voltage')
cooler_cost = S * 1.6
engine_cost = 0.1*cooler_cost + 10
engine_efficiency = 1.0 - abs(1.3 * S - 7.3) - 0.1*(V-220)**2
cooler_water_temperature = 70 - 4.3*S + 0.1 * V

# let's assing names for objectives
engine_cost('engine_cost')
engine_efficiency('engine_efficiency')
cooler_water_temperature('cooler_water_temperature')

objectives = [
     # triplets (objective, tolerance, goal)
     engine_cost, 1, 'min', 
     engine_efficiency, 0.03, 'max', 
     cooler_water_temperature, 1, 40
     ]
     
constraints = [S>0, S<10, V>0, V<300]

startPoint = {S:1.0, V:100} # we could use any start point

p = MOP(objectives, startPoint, constraints = constraints)
r = p.solve('interalg', nProc=1)
''' expected output:
[...]
istop: 1001 (all solutions have been obtained)
Solver:   Time Elapsed = 2.23 	CPU Time Elapsed = 2.23
(on a 2-CPU system with nProc = 2 obtained Time Elapsed = 1.68)
'''
# for solution coordinates and related values of objective functions
# see r.solutions, that is list of points and related objective values with entries like
# {cooler area: 1.0, voltage: 100.0, cooler_water_temperature: 75.700000000000003, engine_cost: 10.16, engine_efficiency: -1445.0}
