from .Polytope import Polytope
class Polyhedron(Polytope): # 3D space polytope
    def __init__(self, *args, **kw):
        Polytope.__init__(self, *args, **kw)
