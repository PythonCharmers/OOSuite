# created by Dmitrey
from numpy import all, ndarray, array, asscalar, asarray, pi, sin, cos
#import numpy as np
from FuncDesigner import  ooarray, dot, sum, sqrt, cross, norm, oofun
from .misc import SpaceFuncsException, pWarn, SF_error
from .baseGeometryObject import baseGeometryObject

pylabInstalled = False
try:
    import pylab
    pylabInstalled = True
except ImportError:
    pass

class Point(ooarray, baseGeometryObject):
    __array_priority__ = 100
    pointCounter = array(0)
    weight = None
    def __init__(self, *args, **kw):
        ooarray.__init__(self)#, *args, **kw)
        baseGeometryObject.__init__(self, *args, **kw)
    def __new__(self, *args, **kw):
        self.pointCounter += 1
        self._num = asscalar(self.pointCounter)
        self.specifier = '.'
        self.color = 'k'
        Args = args[:-1] if type(args[-1]) == str else args
        r =  asarray(Args[0]).view(self) if len(Args) == 1 and type(Args[0]) in (ooarray, Point, ndarray, list, tuple) else ooarray(Args).view(self)
        r.name = args[-1] if type(args[-1]) == str else kw['name'] if 'name' in kw else 'unnamed point ' + str(self._num)
        r._id = oofun._id
        oofun._id += 1
        return r
        #obj = ooarray(asarray(args[:-1] if type(args[-1]) == str else args).tolist()).view(self)
        #return obj

#    def __array_finalize__(self, obj):
#        if obj is None: return

    __call__ = lambda self, *args, **kw: \
    self._name(args[0]) \
    if len(args) == 1 and type(args[0]) == str \
    else Point(ooarray.__call__(self, *args, **kw) if self.size is not 1 or not isinstance(self.view(ndarray)[0], oofun)\
    else asscalar(self)(*args, **kw), weight = self.__dict__.get('weight', None))
       
    _spaceDimension = lambda self: self.size if self.size is not 1 or not isinstance(self.view(ndarray).flatten()[0], oofun) else asscalar(self).size
    
    __getitem__ = lambda self, ind: self.view(ndarray).flatten()[ind] if self.size is not 1 else asscalar(self)[ind]
#    def __get__(self, ind):
#        raise 0
    __getslice__ = lambda self, ind1, ind2: self.view(ndarray)[ind1:ind2] if self.size is not 1 else asscalar(self)[ind1:ind2]
    
    def _name(self, newName):
        self.name= newName
        return self

    __array_wrap__ = lambda self, out_arr, context=None: ndarray.__array_wrap__(self, out_arr, context)

    __radd__ = __add__ = lambda self, other: Point(ooarray.__add__(self, other))
    __rmul__ = __mul__ = lambda self, other: Point(ooarray.__mul__(self, other))
    __div__ = lambda self, other: Point(ooarray.__div__(self, other))
    __rdiv__ = lambda self, other: Point(ooarray.__div__(other, self))

    def __str__(self):
        return str(self.view(ndarray).flatten())
    
    def distance(self, *args, **kw):
        assert len(kw) == 0 and len(args) == 1
        other = args[0]
        if isinstance(other, (list, tuple)):
            other = Point(other)
        if isinstance(other, (Line, Plane)):
            return self.distance(self.projection(other))
        elif isinstance(other, ndarray):
            return sqrt(sum((self-other)**2), attachConstraints = False)
   
    def projection(self, obj):
        assert isinstance(obj, baseGeometryObject), 'incorrect object for projection'
        if isinstance(obj, Line):
            # TODO: remove 0.0 for future Python 3.X versions
            tmp = dot(self-obj.basePoint, obj.direction)
            if isinstance(tmp, oofun): tmp.size = 1
            tmp1 = obj.direction*tmp / (0.0+sum(obj.direction**2))
            tmp2 = obj.basePoint
            if type(tmp1) == ooarray: # hence tmp2 is array
                tmp2 = tmp2.view(ooarray)
            return Point(tmp1+tmp2)
        elif isinstance(obj, Plane):
            d1, d2 = obj.directions
            bp = obj.basePoint
            a = sum(d1**2)
            b = d = dot(d1, d2)
            e = sum(d2**2)
            c = dot(self-bp, d1)
            f =  dot(self-bp, d2)
            delta = a*e-b*d + 0.0 # TODO: remove 0.0 when Python 2.x will be deprecated
            alpha = (c*e-b*f) / delta
            beta = (a*f-c*d)/delta
            return Point(bp + alpha * d1 + beta * d2)
        else:    
            raise SpaceFuncsException('not implemented for this geometry object yet')
    
    def rotate(self, P, angle):
        '''
        Usage: newPoint = originPoint.rotate(BasePoint, angle)
        (angle in radians)
        currently the method is implemented for 2-dimensional space only
        for other spaces you can use FuncDesigner.dot(RotationMatrix ,Point - BasePoint) + BasePoint
        (implementing it with more convenient API is in Future Plans)
        if BasePoint is (0,0), you can just use prevPoint.rotate(0, angle)
        '''
        Tmp = (self - P) if P is not 0 else self
        x, y = Tmp[0], Tmp[1]
        sin_theta, cos_theta = sin(angle), cos(angle)
        x2, y2 = x*cos_theta-y*sin_theta, x*sin_theta+y*cos_theta
        return Point(x2+P[0], y2+P[1]) if P is not 0 else Point(x2, y2)
        
    
    def symmetry(self, other):
        if not isinstance(other, (Point, Line, Plane)):
            raise SpaceFuncsException('method Point.symmetry(object) is implemented for Point|Line|Plane objects only')
        if type(other) == Point:
            return 2*other - self
        else:
            proj = self.projection(other)
            return 2*proj - self
     
    perpendicular = lambda self, obj: \
        _perpendicularToLine(self, obj) if isinstance(obj, Line) \
        else _perpendicularToPlane(self, obj) if isinstance(obj, Plane)\
        else SF_error('perpendicular from Point to the object is not implemented yet')
    
    def plot(self, *args, **kw):
        if not pylabInstalled:
            raise SpaceFuncsException('to plot you should have matplotlib installed')
        # TODO: use something else instead, e.g. scatter
        if self.size != 2:
            raise SpaceFuncsException('only 2-D plot is implemented for now')
        if not self.name.startswith('unnamed'): 
            pylab.plot([self[0]], [self[1]], self.specifier, color=self.color)
            pylab.text(self[0], self[1], self.name)
    #TODO: add FD decision()
    
    # TODO: implement dot from FD

class Line(baseGeometryObject):
    def __init__(self, *args, **kw):
        baseGeometryObject.__init__(self, *args, **kw)
        # TODO: add initialization by SLE
        self.basePoint = args[0] if isinstance(args[0], Point) else Point(args[0])
        if 'direction' in kw:
            self.direction = kw['direction']
        else:
            assert len(args) == 2
            #self.direction = (args[1] if isinstance(args[1], point) else point(args[1])) - self.basePoint
            self.direction = array(args[1]).view(ooarray) - self.basePoint.view(ooarray)#if isinstance(args[1], point) else point(args[1])) - self.basePoint
        if type(self.direction) in (Point, ndarray): self.direction = self.direction.view(ooarray)
        
    __call__ = lambda self, *args, **kw: Line(self.basePoint(*args, **kw), direction = self.direction(*args, **kw))
            
    def __and__(self, other):
        # TODO: attach check for residual < some_val (warn, err)
        assert isinstance(other, Line), 'not implemented for non-line objects'
        points = skewLinesNearestPoints(self, other)
        return 0.5*(points[0] + points[1])
        
    _spaceDimension = lambda self: self.basePoint._spaceDimension()
    
    #perpendicular = lambda self, point: _perpendicularToLine(point, self)
    
    def projection(self, obj):
        assert isinstance(obj, Plane)
        
        # TODO: maybe involve normalization of direction norm?
        r = Line(self.basePoint.projection(obj), (self.basePoint+self.direction).projection(obj))
        r.direction /= norm(r.direction)
        return r
    
    __contains__ = contains = lambda self, point, tol = 1e-6: _contains(self, point, tol)
    
    
class LineSegment(baseGeometryObject):
    # TODO: mb rewrite it with _AttributesDict?
    #_AttributesDict = baseGeometryObject._AttributesDict.copy()
    def __init__(self, Start, End, *args, **kw):
        self.color = 'b'
        assert len(args) == 0
        baseGeometryObject.__init__(self, *args, **kw)
        self.points = [Start, End]
        
    
    __call__ = lambda self, *args, **kw: LineSegment(self.points[0](*args, **kw), self.points[1](*args, **kw))        
    
    _length = lambda self: self.points[0].distance(self.points[1])
    
    _middle = lambda self: 0.5 * (self.points[0] + self.points[1])
    
    def __getattr__(self, attr):
        if attr == 'length':
            self.length = self._length()
            return self.length
        elif attr == 'middle':
            self.middle = self._middle()
            return self.middle
        else:
            raise AttributeError('no such method %s for LineSegment' % attr)
    
    def plot(self):
        if not pylabInstalled: 
            raise SpaceFuncsException('to plot you should have matplotlib installed')
        r = pylab.plot([self.points[0][0], self.points[1][0]], [self.points[0][1], self.points[1][1]], self.color)
        self.points[0].plot()
        self.points[1].plot()
        return r
    
    # TODO: rework it
    _spaceDimension = lambda self: self.points[0]._spaceDimension()

class Plane(baseGeometryObject):
    def __init__(self, *args, **kw):
        baseGeometryObject.__init__(self, *args, **kw)
        self.basePoint = args[0] if isinstance(args[0], Point) else Point(args[0])
        if 'directions' in kw:
            self.directions = kw['directions']
        else:
            assert len(args) == 3
            #self.direction = (args[1] if isinstance(args[1], point) else point(args[1])) - self.basePoint
            self.directions = [array(args[1]).view(ooarray) - (self.basePoint).view(ooarray), 
                                          array(args[2]).view(ooarray) - (self.basePoint).view(ooarray)]
        if isinstance(self.directions[0], Point): self.directions[0] = self.directions[0].view(ooarray)
        if isinstance(self.directions[1], Point): self.directions[1] = self.directions[1].view(ooarray)

    __call__ = lambda self, *args, **kw: Plane(self.basePoint(*args, **kw), directions = [d(*args, **kw) for d in self.directions])

    _spaceDimension = lambda self: self.basePoint._spaceDimension()
    
    __contains__ = contains = lambda self, point, tol = 1e-6: _contains(self, point, tol)


class Ring(baseGeometryObject):
    def __init__(self, center, radius, *args, **kw):
        assert len(args) == 0
        baseGeometryObject.__init__(self, *args, **kw)
        self.center = Point(center)
        self.radius = radius
        self.linewidth = kw.get('linewidth', 1)
        self.linestyle = kw.get('linestyle', 'solid')#['solid' | 'dashed' | 'dashdot' | 'dotted']
        self.edgecolor = kw.get('edgecolor', 'b')
        
        self.fill = kw.get('fill', False)
        # transparency and facecolor are ignored for fill = False
        self.transparency = kw.get('transparency', 0.5)
        self.facecolor = kw.get('facecolor', 'w')
        
        self.plotCenter = True
        self.color = kw.get('color', 'k')
        #self.expected_kwargs |= set(('linewidth', 'linestyle', 'edgecolor', 'fill', 'transparency', 'facecolor'))


    def __getattr__(self, attr):
        if attr in ('S', 'area'): r = self._area() 
        else: raise AttributeError('no such field "%s" in circle instance' % attr)
        setattr(self, attr, r)
        return r
    
    _spaceDimension = lambda self: self.center._spaceDimension()
    
    def plot(self):
        if not pylabInstalled: 
            raise SpaceFuncsException('to plot you should have matplotlib installed')
        cir = pylab.Circle(self.center, radius=self.radius, alpha = 1.0 - self.transparency, lw = self.linewidth, fc='w', fill=self.fill, \
                           ec = self.edgecolor, color = self.color, linestyle = self.linestyle)
        pylab.gca().add_patch(cir)
        if self.plotCenter: self.center.plot()
        
    def _area(self):
        return pi * self.radius ** 2
    
    #contains = __contains__ = lambda self, point, tol = 1e-6: _contains(self, point, tol)

class Circle(Ring):
    def __init__(self, center, radius, *args, **kw):
        assert len(args) == 0
        Ring.__init__(self, center, radius, *args, **kw)
        
        self.fill = kw.get('fill', False)
        # transparency and facecolor are ignored for fill = False

    __call__ = lambda self, *args, **kw: Circle(self.center(*args, **kw) if isinstance(self.center, (oofun, ooarray)) else self.center, \
                                                self.radius(*args, **kw) if isinstance(self.radius, (oofun, ooarray)) else self.radius)

    contains = __contains__ = lambda self, point, tol = 1e-6: _contains(self, point, tol)

    def __getattr__(self, attr):
        if attr == 'disk': 
            r = Disk(self.center, self.radius) 
        elif attr == 'circle':
            r = self
        else: 
            return Ring.__getattr__(self, attr)
        setattr(self, attr, r)
        return r

class Disk(Ring):
    def __init__(self, center, radius, *args, **kw):
        assert len(args) == 0
        Ring.__init__(self, center, radius, *args, **kw)
        
        self.fill = kw.get('fill', True)
        # transparency and facecolor are ignored for fill = False

    __call__ = lambda self, *args, **kw: Disk(self.center(*args, **kw) if isinstance(self.center, (oofun, ooarray)) else self.center, \
                                                self.radius(*args, **kw) if isinstance(self.radius, (oofun, ooarray)) else self.radius)

    contains = __contains__ = lambda self, point, tol = 1e-6: _contains(self, point, tol, inside = True)
    
    def __getattr__(self, attr):
        if attr == 'circle': 
            r = Circle(self.center, self.radius) 
        elif attr == 'disk':
            r = self
        else: 
            return Ring.__getattr__(self, attr)
        setattr(self, attr, r)
        return r


class Orb(baseGeometryObject):
    def __init__(self, center, radius, *args, **kw):
        assert len(args) == 0
        baseGeometryObject.__init__(self, *args, **kw)
        self.center = Point(center)
        self.radius = radius
        
    _area = lambda self: (4 * pi) * self.radius ** 2
    _volume = lambda self: (4.0 / 3 * pi) * self.radius ** 3
    
    def __getattr__(self, attr):
        if attr in ('S', 'area'): r = self._area() 
        elif attr in ('V', 'volume'): r = self._volume() 
        else: raise AttributeError('no such field "%s" in sphere instance' % attr)
        setattr(self, attr, r)
        return r
    
    #__contains__ = contains = lambda self, point, tol = 1e-6: _contains(self, point, tol)

class Sphere(Orb):
    __call__ = lambda self, *args, **kw: Sphere(self.center(*args, **kw) if isinstance(self.center, (oofun, ooarray)) else self.center, \
                                                self.radius(*args, **kw) if isinstance(self.radius, (oofun, ooarray)) else self.radius)
    __contains__ = contains = lambda self, point, tol = 1e-6: _contains(self, point, tol)
    def __getattr__(self, attr):
        if attr == 'ball': 
            r = Ball(self.center, self.radius) 
        elif attr == 'shpere':
            r = self
        else: 
            return Orb.__getattr__(self, attr)
        setattr(self, attr, r)
        return r


class Ball(Orb):
    __call__ = lambda self, *args, **kw: Ball(self.center(*args, **kw) if isinstance(self.center, (oofun, ooarray)) else self.center, \
                                                self.radius(*args, **kw) if isinstance(self.radius, (oofun, ooarray)) else self.radius)
    __contains__ = contains = lambda self, point, tol = 1e-6: _contains(self, point, tol, inside = True)
    def __getattr__(self, attr):
        if attr == 'sphere': 
            r = Sphere(self.center, self.radius) 
        elif attr == 'ball':
            r = self
        else: 
            return Orb.__getattr__(self, attr)
        setattr(self, attr, r)
        return r
    

def skewLinesNearestPoints(line1, line2):
    assert isinstance(line1, Line) and isinstance(line2, Line)
    p1, p2 = line1.basePoint, line2.basePoint
    d1, d2 = line1.direction, line2.direction
    Delta = sum(d1**2)*sum(d2**2) - dot(d1, d2)**2
    Delta1 = dot(d2, p1-p2) * dot(d1, d2) - dot(d1, p1-p2) * sum(d2**2)
    Delta2 = dot(d2, p1-p2) * sum(d1**2) - dot(d1, d2) * dot(d1, p1-p2)
    t1 = Delta1 / Delta
    t2 = Delta2 / Delta
    for t in [t1, t2]:
        if isinstance(t, oofun): t.size = 1
    return Point(p1+t1*d1), Point(p2+t2*d2)
    
def _perpendicularToLine(point, line): #, **kw):
    ############################################
    # Don't change "is" by "=="!!!
    if point.size is 2 or line.basePoint.size is 2 or line.direction.size is 2:
        # thus 2D space is involved
        return Line(point, direction=ooarray(line.direction[1], -line.direction[0]))
    ############################################
    projection = point.projection(line)
    if projection.dtype != object and all(projection.view(ndarray)==point.view(ndarray)):
        if 'size' not in dir(point) + dir(line.basePoint) + dir(line.direction):
            s = '''
            The point belongs to the line, hence
            to perform the operation safely you should provide 
            space dimension (as "size" parameter for 
            either point or line point or line direction).
            Assuming space dimension is 2 
            (elseware lots of perpendicular lines wrt the point and the line exist)
            '''
            pWarn(s)
        else:
            raise SpaceFuncsException('for space dimension > 2 you should have point outside of the line')
        return Line(point, direction=ooarray(line.direction[1], -line.direction[0]))
    else:
        return Line(point, projection)

# TODO: add unknown spaceDimension warning
_perpendicularToPlane=lambda point, plane:\
    Line(point, direction=cross(*plane.directions)) \
    if point.size is 3 or plane.basePoint.size is 3 or plane.directions[1].size is 3 or plane.directions[1].size is 3\
    else Line(point, point.projection(plane))
    
#TODO: intersection of plane & line


def _contains(obj, p, tol, inside = False):
    assert isinstance(p, Point), 'implemented only for Point yet'
    if isinstance(obj, (Line, Plane)):
        P = p.projection(obj)
        if isinstance(P, ndarray) and str(p.dtype) != 'object':
            return norm(p - P) <= tol
        else:
            return (p == P)(tol=tol)
    elif isinstance(obj, (Ring, Orb)):
        if isinstance(p, ndarray) and str(p.dtype) != 'object':
            return p.distance(obj.center) - obj.radius <= tol if inside else -tol <= p.distance(obj.center) - obj.radius <= tol
        else:
            return (p.distance(obj.center) <= obj.radius if inside else p.distance(obj.center) == obj.radius)(tol=tol)
    else:
        assert 0, 'method contains() is unimplemented for type %s' % type(obj)
        
