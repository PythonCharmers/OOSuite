from .Polytope import Polytope
from FuncDesigner import ooarray, sqrt, angle, abs, sum
from numpy import arange, isscalar
from .misc import SpaceFuncsException

pylabInstalled = False
try:
    import pylab
    pylabInstalled = True
except ImportError:
    pass
    
table = {
         'p': 'semiperimeter',
         'P': 'perimeter', 
         'S': 'area', 
}
table = dict([(key, '_'+val) for key, val in table.items()] + [(val, '_'+val) for key, val in table.items()])
others = ['sides', 'angles', 'area']
table.update([(s, '_'+s) for s in others])

class Polygon(Polytope):
    _AttributesDict = Polytope._AttributesDict.copy()
    def __init__(self, *args, **kw):
        Polytope.__init__(self, *args, **kw)
        
    _sides = lambda self: ooarray([sqrt(sum((p1-p2)**2), attachConstraints=False) \
                                             for p1, p2 in \
                                             [(self.vertices[i], self.vertices[i+1]) for i in range(self.nVertices-1)] \
                                             + [(self.vertices[self.nVertices-1], self.vertices[0])]])
    
    _perimeter = lambda self: sum(self.sides)
    _semiperimeter = lambda self: 0.5*self.P
    
    _angles = lambda self: ooarray([angle(p1, p2) for p1, p2 in \
                                             [(self.vertices[i-1]-self.vertices[i], self.vertices[i+1]-self.vertices[i]) for i in range(self.nVertices-1)]+\
                                             [(self.vertices[self.nVertices-2]-self.vertices[self.nVertices-1], self.vertices[0]-self.vertices[self.nVertices-1])]])

    def _area(self):
        D = self._spaceDimension()
        if isscalar(D) and D != 2:
            raise SpaceFuncsException('polygon area is not implemented for space dimension > 2 yet')
        x, y = self._coords(0), self._coords(1)
        x.append(x[0])
        y.append(y[0])
        x, y = ooarray(x), ooarray(y)
        return 0.5*abs(sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
    
    
    def plot(self, *args, **kw):
        if not pylabInstalled: raise SpaceFuncsException('to plot you should have matplotlib installed')
        #pylab.Line2D.__init__([self.vertices[i][0] for i in range(3)], [self.vertices[i][1] for i in range(3)])
        #raise 0
        pylab.plot([self.vertices[i][0] for i in (arange(self.nVertices).tolist() + [0])], 
                        [self.vertices[i][1] for i in (arange(self.nVertices).tolist() + [0])], 
                        *args, **kw)
        for i in range(3): 
            self.vertices[i].plot()
        pylab.draw()

    __call__ = lambda self, *args, **kw: Polygon([self.vertices[i](*args, **kw) for i in range(self.nVertices)])


Polygon._AttributesDict.update(table)
