'''
An example of optimization in SpaceFuncs with olympic rings
Let's search 5 points (1 from each ring) that form polygon with maximum area
'''
from SpaceFuncs import *
from FuncDesigner import *
from openopt import *
from numpy import linspace

centers = [(-2, 0), (0, 0), (2, 0), (1, -1), (-1, -1)]
colors = ['b', 'k', 'r', 'g', 'y']
radius = 0.9
circles = [Circle(center, radius, edgecolor = colors[i], linewidth = 3) for i, center in enumerate(centers)]

n = len(circles)
x = oovars(n)
y = oovars(n)

Points = [Point(x[i], y[i]) for i in range(n)]
P = Polygon(*Points)
objective = P.S # area

# we will deal with disks instead of circles because handling nonlinear equalities is more difficult then inequalities
# you can switch via d = circle.disk, c = disk.circle (without brackets)
constraints = [circles[i].disk.contains(Points[i], tol = 1e-5) for i in range(n)]

X, Y = linspace(-1, 1, n), [0]*n
startPoint = {x: X, y: Y}

p = NLP(objective, startPoint, constraints = constraints)

solver = 'scipy_cobyla' # requires SciPy installed; see openopt.org for other available solvers
p.debug = 1
r = p.maximize(solver)
''' Intel Atom 1.6 GHz result:
------------------------- OpenOpt 0.39 -------------------------
solver: scipy_cobyla   problem: unnamed    type: NLP   goal: maximum
 iter   objFunVal   log10(MaxResidual/ConTol)   
    0  0.000e+00                      4.22 
   10  1.063e+01                      2.08 
   20  1.069e+01                      0.64 
   30  1.068e+01                     -5.60 
   40  1.068e+01                     -8.15 
istop: 1000
Solver:   Time Elapsed = 5.0 	CPU Time Elapsed = 4.99
objFunValue: 10.684673 (feasible, max(residuals/requiredTolerances) = 7.09988e-09)
'''

# some basic plotting is available in SpaceFuncs API:
for circle in circles:
    circle.plot()

# For features beyond it use matplotlib: 
import pylab
xx, yy = x(r).tolist(), y(r).tolist()
xx.append(xx[0])
yy.append(yy[0])
pylab.plot(xx, yy, marker = (5, 1, 0), markersize = 15)
pylab.xlim([-3, 3])
pylab.ylim([-3, 3])
pylab.show()
# see http://openopt.org/Image:olympic.png 
