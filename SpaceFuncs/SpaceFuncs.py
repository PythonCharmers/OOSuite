# created by Dmitrey
import os,sys
Sep = os.sep
sfPath = ''.join(elem+Sep for elem in __file__.split(Sep)[:-1])
sys.path.append(sfPath + 'kernel')
from baseObjects import Point, Line, LineSegment, Plane,  Circle, Disk, skewLinesNearestPoints, Sphere, Ball
for name in ['Triangle', 'Tetrahedron', 'Polygon', 'Polytope']:
    exec('from %s import %s' % (name, name))
#from kernel.Triangle import Triangle
from kernel.Plot import Plot
from numpy import pi
