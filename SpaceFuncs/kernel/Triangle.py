# created by Dmitrey
from FuncDesigner import sqrt, sum, ooarray, angle  #, stack
from .baseObjects import Line, Circle
#from misc import SpaceFuncsException
from .Polygon import Polygon

table = {
         'r': 'inscribedCircleRadius', 
         'R': 'circumCircleRadius', 
         'H': 'orthocenter', 
         'O': 'circumCircleCenter', 
         'I': 'incenter'
         }
table = dict([(key, '_'+val) for key, val in table.items()] + [(val, '_'+val) for key, val in table.items()])
others = ['CircumCircle', 'InscribedCircle']
table.update([(s, '_'+s) for s in others])

class Triangle(Polygon):
    #vertices = None
    nVertices = 3
    _AttributesDict = Polygon._AttributesDict.copy()
    def __init__(self, *args, **kw):
        assert len(kw) == 0 and len(args) in (1, 3)
        Polygon.__init__(self, *args, **kw)
        
    __call__ = lambda self, *args, **kw: Triangle([self.vertices[i](*args, **kw) for i in range(self.nVertices)])        

    _sides = lambda self: ooarray([sqrt(sum((p1- p2)**2), attachConstraints=False) for p1, p2 in \
                                    ((self.vertices[1], self.vertices[2]), 
                                    (self.vertices[2], self.vertices[0]), 
                                    (self.vertices[0], self.vertices[1]))])

    _angles = lambda self: ooarray([angle(p1, p2) for p1, p2 in \
                                    ((self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]), 
                                    (self.vertices[2]-self.vertices[1], self.vertices[0]-self.vertices[1]), 
                                    (self.vertices[0]-self.vertices[2], self.vertices[1]-self.vertices[2]))])
    
    _area = lambda self: Polygon._area(self) if self._spaceDimension() is 2 else self._areaViaGeron() 
    #_area = lambda self: Polygon._area(self) if self._spaceDimension() is 2 else self._areaViaHeightSideMult() 
    
    _areaViaGeron = lambda self: sqrt(self.p * (self.p - self.sides[0]) * (self.p - self.sides[1]) * (self.p - self.sides[2]), attachConstraints = False)
    
    def _areaViaHeightSideMult(self):
        proj = self.vertices[0].projection(Line(self.vertices[1], self.vertices[2]))
        return 0.5 * proj.distance(self.vertices[0]) * self.vertices[1].distance(self.vertices[2])
   
    _circumCircleRadius = lambda self: (self.sides[0] * self.sides[1] * self.sides[2]) / (4*self.S)
    
    _inscribedCircleRadius = lambda self: self.S / self.p
   
    _incenter = lambda self: (self.vertices[0]*self.sides[0] + self.vertices[1]*self.sides[1] + self.vertices[2]*self.sides[2]) / self.P

    _orthocenter = lambda self: self.vertices[0].perpendicular(Line(self.vertices[1], self.vertices[2])) \
    & self.vertices[1].perpendicular(Line(self.vertices[0], self.vertices[2]))
    
    # Eiler's theorem: 2 * vector(OM) = vector(MH) => O = M -0.5 MH = M - 0.5(H-M) = 1.5M - 0.5H
    _circumCircleCenter = lambda self: 1.5*self.M - 0.5*self.H
    
    _InscribedCircle = lambda self: Circle(self.I, self.r)
    _CircumCircle = lambda self: Circle(self.O, self.R)


Triangle._AttributesDict.update(table)

