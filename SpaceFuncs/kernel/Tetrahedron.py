# created by Dmitrey
#from numpy import array, asscalar, isscalar, ndarray
from FuncDesigner import dot, cross, sum, det3, norm #, sqrt, sum, oofun, ooarray, anglet  #, stack
#from BaseObjects import Point, Line, Circle
from .baseObjects import Sphere
#from misc import SpaceFuncsException
from .Polyhedron import Polyhedron

table = {
         'S': 'square', 
         'I': 'incenter',          
         'r': 'insphereRadius', 
         'O': 'circumSphereCenter',          
         'R': 'circumSphereRadius', 
#         'H': 'orthocenter', 
         'V': 'volume'
         }
table = dict([(key, '_'+val) for key, val in table.items()] + [(val, '_'+val) for key, val in table.items()])
others = ['CircumSphere', 'InscribedSphere', 'reducedVertices', 'reducedVerticesCrossProduct']
table.update([(s, '_'+s) for s in others])


# TODO: add angles
class Tetrahedron(Polyhedron): # TODO: derive from polytop instead of _baseGeometryObject
    nVertices = 4
    _AttributesDict = Polyhedron._AttributesDict.copy()
    def __init__(self, *args, **kw):
        assert len(kw) == 0 and len(args) in (1, 4)
        Polyhedron.__init__(self, *args, **kw)

    __call__ = lambda self, *args, **kw: Tetrahedron([self.vertices[i](*args, **kw) for i in range(self.nVertices)])

    def _volume(self): 
        a, b, c = self.reducedVertices
        return det3(a, b, c) / 6.0
#    
    _insphere = lambda self: Sphere(self.insphereCenter, self.insphereRadius)

    def _reducedVertices(self):
        a, b, c, d = self.vertices
        return a-d, b-d, c-d
    
    def _reducedVerticesCrossProduct(self):
        a, b, c = self.reducedVertices
        return cross(b, c), cross(c, a), cross(a, b)

    def _insphereRadius(self):
        a, b, c = self.reducedVertices
        bc, ca, ab = self.reducedVerticesCrossProduct
        return 6.0*self.V / (norm(bc) + norm(ca) + norm(ab) + norm(bc+ca+ab))
#    
    def _circumSphereRadius(self):
        a, b, c = self.reducedVertices
        bc, ca, ab = self.reducedVerticesCrossProduct
        return norm(sum(a**2)*bc + sum(b**2)*ca + sum(c**2)*ab) / (12.0 * self.V)
        
    def _incenter(self):
        a, b, c = self.reducedVertices
        bc, ca, ab = self.reducedVerticesCrossProduct
        return (a * norm(bc) + b * norm(ca) + c * norm(ab)) / (norm(bc)+norm(ca)+norm(ab)+norm(bc+ca+ab)) + self.vertices[-1]

    def _circumSphereCenter(self):
        a, b, c = self.reducedVertices
        bc, ca, ab = self.reducedVerticesCrossProduct
        return (sum(a**2) * bc + sum(b**2) * ca + sum(c**2) * ab) / (2.0*dot(a, bc)) + self.vertices[-1]
    
    _CircumSphere = lambda self: Sphere(self.circumSphereCenter, self.circumSphereRadius)
    
    _Insphere = lambda self: Sphere(self.incenter, self.insphereRadius)
    
    #_sides = lambda self: ooarray([sqrt(sum((p1-p2)**2), attachConstraints=False) for p1, p2 in ((self.vertices[1], self.vertices[2]), (self.vertices[2], self.vertices[0]), (self.vertices[0], self.vertices[1]))])
    #_sides = lambda self: ooarray([sqrt(sum((p1-p2)**2), attachConstraints=False) for p1, p2 in ((self.vertices[1], self.vertices[2]), (self.vertices[2], self.vertices[0]), (self.vertices[0], self.vertices[1]))])
    
    #_angles = lambda self: ooarray([angle(p1, p2) for p1, p2 in ((self.vertices[1], self.vertices[2]), (self.vertices[2], self.vertices[0]), (self.vertices[0], self.vertices[1]))])

    #_perimeter = lambda self: sum(self.sides)
    
    #_semiperimeter = lambda self: 0.5*sum(self.sides)
    
    #_square = lambda self: sqrt(self.p * (self.p - self.sides[0]) * (self.p - self.sides[1]) * (self.p - self.sides[2]), attachConstraints = False)
    
    #_inscribedCircleRadius = lambda self: sqrt((self.p - self.sides[0]) * (self.p - self.sides[1]) * (self.p - self.sides[2]) / self.p, attachConstraints = False)
    
    #_circumCircleRadius = lambda self: (self.sides[0] * self.sides[1] * self.sides[2]) / (4*self.S)
    
    #_inscribedCircleRadius = lambda self: self.S / self.p

    #_centroid = lambda self: (self.vertices[0] + self.vertices[1] + self.vertices[2]) / 3.0
    
    #_incenter = lambda self: (self.vertices[0]*self.sides[0] + self.vertices[1]*self.sides[1] + self.vertices[2]*self.sides[2]) / self.P

    #_orthocenter = lambda self: self.vertices[0].perpendicular(line(self.vertices[1], self.vertices[2])) \
    #& self.vertices[1].perpendicular(line(self.vertices[0], self.vertices[2]))
    
    # Eiler's theorem: 2 * vector(OM) = vector(MH) => O = M -0.5 MH = M - 0.5(H-M) = 1.5M - 0.5H
    #_circumCircleCenter = lambda self: 1.5*self.M - 0.5*self.H
    
    #_InscribedSphere = lambda self: Sphere(self.I, self.r)
    #_CircumSphere = lambda self: Sphere(self.O, self.R)
    
Tetrahedron._AttributesDict.update(table)
