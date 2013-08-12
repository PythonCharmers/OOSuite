# created by Dmitrey

from kernel.baseObjects import Point, Line, LineSegment, Plane,  Circle, Disk, skewLinesNearestPoints, Sphere, Ball
for name in ['Triangle', 'Tetrahedron', 'Polygon', 'Polytope', 'Plot']:
    exec('from kernel.%s import %s' % (name, name))
from numpy import pi
