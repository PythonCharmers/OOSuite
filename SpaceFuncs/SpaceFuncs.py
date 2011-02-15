# created by Dmitrey
from kernel.baseObjects import Point, Line, LineSegment, Plane,  Circle, skewLinesNearestPoints
for name in ['Triangle', 'Tetrahedron', 'Polygon', 'Polytope']:
    exec('from kernel.%s import %s' % (name, name))
#from kernel.Triangle import Triangle
from kernel.Plot import Plot
