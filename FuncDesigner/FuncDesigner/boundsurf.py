PythonSum = sum
import numpy as np
from operator import le as LESS,  gt as GREATER

class surf:
    def __init__(self, d, c):
        self.d = d # dict of variables and linear coefficients on them (probably as multiarrays)
        self.c = c # (multiarray of) constant(s)
        #self.Type = Type # False for lower, True for upper

    def resolve(self, domain, cmp):
        r = PythonSum(np.where(cmp(v, 0), domain[k][0], domain[k][1])*v for k, v in self.d.items())
        return r + self.c
    
#    def __getattr__(self, attr):
#        if attr == 'resolve_index':
#            assert 0, 'resolve_index must be used from surf derived classes only'
#        else:
#            raise AttributeError('error in FD engine (class surf)')
            
    def __add__(self, other):
        if type(other) == type(self):
            #assert self.__class__ == other.__class__, 'bug in FD kernel (class surf)'
            S, O = self.d, other.d
            d = S.copy()
            d.update(O)
            for key in set(S.keys()) & set(O.keys()):
                d[key] = S[key]  + O[key]
            return surf(d, self.c+other.c)
        elif np.isscalar(other) or (type(other) == np.ndarray):
            return surf(self.d, self.c + other)
        else:
            assert 0, 'unimplemented yet'
    
    def __mul__(self, other):
        if np.isscalar(other) or (type(other) == np.ndarray and other.size == 1):
            return surf(dict([(k, v*other) for k, v in self.d.items()]), self.c * other)
#        elif type(other) == surf:
#            return surf(self.l+other.l, self.u+other.u)
        else:
            assert 0, 'unimplemented yet'


class boundsurf:
    __array_priority__ = 15
    def __init__(self, lowersurf, uppersurf, definiteRange):
        #print('b')
        self.l = lowersurf
        self.u = uppersurf
        self.definiteRange = definiteRange
        
    def resolve(self, domain):
        l = self.l.resolve(domain, GREATER)
        u = self.u.resolve(domain, LESS)
        return np.vstack((l, u)), self.definiteRange
        
    def isfinite(self):
        return np.all(np.isfinite(self.l.c)) and np.all(np.isfinite(self.u.c))
        
    # TODO: handling fd.sum()
    def __add__(self, other):
        if np.isscalar(other) or (type(other) == np.ndarray and other.size == 1):
            return boundsurf(self.l+other, self.u+other, self.definiteRange)
        elif other.__class__ == boundsurf:# TODO: replace it by type(r[0]) after dropping Python2 support
            return boundsurf(self.l+other.l, self.u+other.u, self.definiteRange & other.definiteRange)
        elif type(other) == np.ndarray:
            assert other.shape[0] == 2, 'unimplemented yet'
            return boundsurf(self.l+other[0], self.u+other[1], self.definiteRange)
        else:
            assert 0, 'unimplemented yet'
            
    __radd__ = lambda self, other: self.__add__(other)
    
    def __neg__(self):
        l, u = self.l, self.u
        L = surf(dict([(k, -v) for k, v in u.d.items()]), -u.c)
        U = surf(dict([(k, -v) for k, v in l.d.items()]), -l.c)
        return boundsurf(L, U, self.definiteRange)
    
    # TODO: mb rework it
    __sub__ = lambda self, other: self + (-other)
        
    def __mul__(self, other):
        if np.isscalar(other) or (type(other) == np.ndarray and other.size == 1):
            if other >= 0:
                return boundsurf(self.l*other, self.u*other, self.definiteRange)
            else:
                return boundsurf(self.u*other, self.l*other, self.definiteRange)
#        elif other.__class__ == boundsurf:
#            return boundsurf(self.l+other.l, self.u+other.u)
        else:
            assert 0, 'unimplemented yet'
            
    # TODO: rework it if __iadd_, __imul__ etc will be created
    def copy(self):
        return self
        
