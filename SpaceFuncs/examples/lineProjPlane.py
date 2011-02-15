from SpaceFuncs import *

# 3D example
p1, p2, p3 = Point(1, 0, 0), Point(0, 2, 0), Point(0, 0, 4)
plane = Plane(p1, p2, p3)

P1, P2 = Point(0, 0, 0), Point(0, 0, 1)
line = Line(P1, P2)

projectedLine = line.projection(plane)

