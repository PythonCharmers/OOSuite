#!/usr/bin/python
'''
A simple OpenOpt multiobjective TSP example for directed multigraph using interalg solver;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/TSP for more details
'''
from openopt import *
from numpy import sin, cos#, abs

import networkx as nx
N = 5
G = nx.MultiDiGraph()

G.add_edges_from(\
                 [(i,j,{'time': 1.5*(cos(i)+sin(j)+1)**2, 'cost':(i-j)**2 + 2*sin(i) + 2*cos(j)+1, 'way': 'aircraft'}) for i in range(N) for j in range(N) if i != j ])

G.add_edges_from(\
                 [(i,j,{'time': 4.5*(cos(i)-sin(j)+1)**2, 'cost':(i-j)**2 + sin(i) + cos(j)+1, 'way': 'railroad'}) for i in range(int(2*N/3)) for j in range(int(N)) if i != j ])

G.add_edges_from(\
                 [(i,j,{'time': 4.5*(cos(4*i)-sin(3*j)+1)**2, 'cost':(i-2*j)**2 + sin(10+i) + cos(2*j)+1, 'way': 'aircraft'}) for i in range(int(2*N/3)) for j in range(int(N)) if i != j ])

G.add_edges_from(\
                 [(i,j,{'time': +(4.5*(cos(i)+cos(j)+1)**2 + abs(i - j)), 'cost': i + j**2, 'way': 'railroad'}) for i in range(int(N)) for j in range(int(N)) if i != j ])

objective = [
              # name, tol, goal
              'time', 0.005, 'min', 
              'cost', 0.005, 'min'
              ]

# for solver sa handling of constraints is unimplemented yet
# when your solver is interalg you can use nonlinear constraints as well
# for nonlinear objective and constraints functions like arctan, abs etc should be imported from FuncDesigner
from FuncDesigner import arctan
constraints = lambda value: (
                             2 * value['time'] + 3 * value['cost'] ** 2 > 100, 
                             8 * arctan(value['time']) + 15*value['cost'] > 150
                             )


p = TSP(G, objective = objective, constraints = constraints)


r = p.solve('interalg', nProc=2) # see http://openopt.org/interalg for more info on the solver

# you can provide some stop criterion, 
# e.g. maxTime, maxCPUTime, fEnough etc, for example 
# r = p.solve('interalg', maxTime = 100, maxCPUTime=100) 

# also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving) 
# it requires tkinter installed, that is included into PythonXY, EPD;
# for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
#r = p.manage('interalg')

print(r.solutions.values)
print(r.solutions)
'''
Solver:   Time Elapsed = 23.5 	CPU Time Elapsed = 21.71
2 solutions have been obtained
array([[ 57.58275032,  25.1063021 ],
       [ 48.37229463,  41.89946006]])
[[(0, 2, {'cost': 15.802335268247019, 'way': 'aircraft', 'time': 23.380807560432558}), 
(1, 0, {'cost': 3.8414709848078967, 'way': 'railroad', 'time': 10.676390370582189}), 
(2, 4, {'cost': 5.2556538059620701, 'way': 'railroad', 'time': 8.0881091791529247}), 
(3, 1, {'cost': 4, 'way': 'railroad', 'time': 3.3627839877931076}), 
(4, 3, {'cost': 13, 'way': 'railroad', 'time': 2.864203532668927})], 
[(0, 1, {'cost': 1, 'way': 'railroad', 'time': 30.039111123395447}), 
(1, 2, {'cost': 2.8506482965215083, 'way': 'aircraft', 'time': 9.0008082756204608}), 
(2, 4, {'cost': 5.2556538059620701, 'way': 'railroad', 'time': 8.0881091791529247}), 
(3, 0, {'cost': 3, 'way': 'railroad', 'time': 7.5905182061553056}), 
(4, 3, {'cost': 13, 'way': 'railroad', 'time': 2.864203532668927})]]
'''
