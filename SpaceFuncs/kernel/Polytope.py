from .baseGeometryObject import baseGeometryObject
from .baseObjects import Point
from numpy import ndarray
from FuncDesigner import ooarray, sum

table = {'M':'centroid'}
table = dict([(key, '_'+val) for key, val in table.items()] + [(val, '_'+val) for key, val in table.items()])

class Polytope(baseGeometryObject):
    nVertices = 0 # unknown
    _AttributesDict = baseGeometryObject._AttributesDict.copy()
    def __init__(self, *args, **kw):
        baseGeometryObject.__init__(self, **kw)
        if len(args)==1:
            assert type(args[0]) in (list, tuple, ndarray, ooarray)
            self.vertices = [Point(arg) for arg in args[0]] 
        else:
            self.vertices = [args[i] if type(args[i]) == Point \
                                                   else Point(args[i]) for i in range(len(args))]
        self.nVertices = len(self.vertices)

    #_centroid = lambda self: sum(self.vertices)/ float(self.nVertices)
    def _centroid(self):
        tmp = [v.weight is not None for v in self.vertices]
        if all(tmp):
            return sum([v*v.weight for  v in self.vertices]) / sum([v.weight for v in self.vertices])
        elif not any(tmp):
            return sum(self.vertices)/ float(self.nVertices)
        else:
            assert 0, 'to get centroid you should either provide weight for all vertices or for noone'
    
    def _spaceDimension(self):
        # TODO: rework it
        return self.vertices[0]._spaceDimension()

    def _coords(self, ind):
        return [self.vertices[i][ind] for i in range(self.nVertices)]
    
    __call__ = lambda self, *args, **kw: Polytope([self.vertices[i](*args, **kw) for i in range(self.nVertices)])
    
Polytope._AttributesDict.update(table)
