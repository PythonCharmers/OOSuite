from SpaceFuncs import *

# 2D example
p1, p2 = Point(0, 2), Point(4, 0)
line = Line(p1, p2)

P1, P2, P3 = Point(3.9, 0), Point(0, 2.1), Point(0, 0)

print('P1 projection: ' + str(P1.projection(line))) # [ 3.92  0.04]
print('P2 projection: ' + str(P2.projection(line))) # [-0.04  2.02]
print('P3 projection: ' + str(P3.projection(line))) # [ 0.8  1.6]

# Let's form a line that includes a P3 and is perpendicular to the Line:
line2 = P3.perpendicular(line)
# currently to form the perpendicular line
# point can belong to the original line 
# if and only if 2D space is involved

""" example of points projection with parameters in 5D space """
from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c')
p1 = Point(0, 2, 3, 4, a)
p2 = Point(4, 0, 3, b, c)
line = Line(p1, p2)
P = Point(1, 2+b*c, 3*a, 4+exp(b), a*sqrt(c))
proj = P.projection(line)
Parameters = {a:1, b:2, c:0.5}
print('for parameters %s projection is %s' % (Parameters,  proj(Parameters)))
# for parameters {a: 1, b: 2, c: 0.5} projection is [-2.08357371  3.04178685  3.  5.04178685   1.26044671]
