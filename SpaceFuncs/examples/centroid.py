'''
Suppose we have several points with different positive or negative weights and would like to place them in the olympic circles
so that their centroid (barycenter) would be as close to a target as possible; let's also demand distance between any 2 points 
has to be no less than 0.15. It is combinatorial optimization problem - each point may belong to one of 5 circles, 
so number of possibilities is 5^m, where m is number of points; moreover, we have m*(m-1)/2 difficult nonlinear nonconvex equality constraints 
(requirements on distances). Fortunately, OpenOpt has awesome solver interalg, that easily solves this problem (for quite small m, although); 
we will consider 3 points of weights [15, -0.15, 3].
'''
from SpaceFuncs import *
from FuncDesigner import *
from openopt import *
from numpy import linspace

Target = Point(1.15, 1.5)

weights = [15, -0.15, 3]
m = len(weights)

centers = [(-2, 0), (0, 0), (2, 0), (1, -1), (-1, -1)]
colors = ['b', 'k', 'r', 'g', 'y']
radius = 0.9
circles = [Circle(center, radius, edgecolor = colors[i], linewidth = 3) for i, center in enumerate(centers)]
n = len(circles)

x = oovars(m)
y = oovars(m)

Points = [Point(x[i], y[i], weight = weights[i]) for i in range(m)]
P = Polygon(*Points)
Centroid = P.centroid # barycenter
objective = Centroid.distance(Target)

# each point must be included into one of circles
constraints = [OR([circles[i].contains(Point, tol = 1e-3) for i in range(n)]) for Point in Points]
# suppose distance between any 2 points has to be no less than 0.15
constraints += [Points[i].distance(Points[j]) > 0.15 for i in range(m) for j in range(i+1, m)]
X, Y = linspace(-1, 1, m), [0]*m
startPoint = {x: X, y: Y}

fTol = 0.01 # required tolerance |f - f*| <= fTol
implicitBounds = (-3, 3) # lb <= x <= ub for those solvers where finite box bounds are required
# you can use mere "implicitBounds = 3" here
p = NLP(objective, startPoint, constraints = constraints, implicitBounds = implicitBounds, fTol = fTol)

solver = 'interalg' 
r = p.solve(solver, iprint = 50)
''' Intel Atom 1.6 GHz result:
------------------------- OpenOpt 0.39 -------------------------
solver: interalg   problem: unnamed    type: NLP   goal: minimum
 iter   objFunVal   log10(maxResidual)   
    0  2.360e+00               0.00 
   50  2.189e+00            -100.00 
  100  7.127e-01            -100.00 
  150  5.900e-01            -100.00 
OpenOpt info: Solution with required tolerance 1.0e-02 
 is guarantied (obtained precision: 1.0e-02)
  156  5.900e-01            -100.00 
istop: 1000 (solution has been obtained)
Solver:   Time Elapsed = 75.85 	CPU Time Elapsed = 74.48
objFunValue: 0.58996593 (feasible, MaxResidual = 0)
'''

# some basic plotting is available in SpaceFuncs API:
for circle in circles:
    circle.plot()

# For features beyond it use matplotlib: 
import pylab
pylab.scatter(x(r), y(r), color='b', s = 80)
pylab.scatter(Target[0], Target[1], color='r', marker = (5, 1, 0), s = 80)
solutionCentroid = Centroid(r)
pylab.scatter(solutionCentroid[0], solutionCentroid[1], color='b', marker = (5, 1, 0), s = 80)
pylab.xlim([-3, 3])
pylab.ylim([-3, 3])
pylab.show()
# see http://openopt.org/Image:centroid.png 
