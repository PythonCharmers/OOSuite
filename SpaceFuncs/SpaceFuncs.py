# created by Dmitrey
import os,sys
Sep = os.sep
sfPath = ''.join(elem+Sep for elem in __file__.split(Sep)[:-1])
sys.path.append(sfPath + 'kernel')
from kernel.baseObjects import Point, Line, LineSegment, Plane,  Circle, skewLinesNearestPoints, Sphere
for name in ['Triangle', 'Tetrahedron', 'Polygon', 'Polytope']:
    exec('from kernel.%s import %s' % (name, name))
#from kernel.Triangle import Triangle
from kernel.Plot import Plot
from numpy import pi
