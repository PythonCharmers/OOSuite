PythonSum = sum
import numpy as np
from operator import le as LESS,  gt as GREATER

class surf:
    isRendered = False
    def __init__(self, d, c):
        self.d = d # dict of variables and linear coefficients on them (probably as multiarrays)
        self.c = c # (multiarray of) constant(s)
        #self.Type = Type # False for lower, True for upper

    def resolve(self, domain, cmp):
        r = PythonSum(np.where(cmp(v, 0), domain[k][0], domain[k][1])*v for k, v in self.d.items())
        return r + self.c
    
    def render(self, domain, cmp):
        self.rendered = dict([(k, np.where(cmp(v, 0), domain[k][0], domain[k][1])*v) for k, v in self.d.items()])
        self.resolved = PythonSum(self.rendered) + self.c
        self.isRendered = True
    
    def __add__(self, other):
        if other.__class__ ==surf:
            if other.isRendered and not self.isRendered:
                self, other = other, self
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
            
    __rmul__ = lambda self, other: self.__mul__(other)
            
#    def __getattr__(self, attr):
#        if attr == 'resolve_index':
#            assert 0, 'resolve_index must be used from surf derived classes only'
#        else:
#            raise AttributeError('error in FD engine (class surf)')
            

class boundsurf:
    __array_priority__ = 15
    isRendered = False
    def __init__(self, lowersurf, uppersurf, definiteRange, domain):
        #print('b')
        self.l = lowersurf
        self.u = uppersurf
        self.definiteRange = definiteRange
        self.domain = domain
        
    def Size(self):
        return max((len(self.l.d), len(self.u.d), 1))
        
    def resolve(self):
        l = self.l.resolve(self.domain, GREATER)
        u = self.u.resolve(self.domain, LESS)
        return np.vstack((l, u)), self.definiteRange
    
    def render(self):
        if self.isRendered:
            return
        self.l.render(self, self.domain, GREATER)
        self.u.render(self, self.domain, LESS)
        self.isRendered = True
    
    def isfinite(self):
        return np.all(np.isfinite(self.l.c)) and np.all(np.isfinite(self.u.c))
        
    # TODO: handling fd.sum()
    def __add__(self, other):
        if np.isscalar(other) or (type(other) == np.ndarray and other.size == 1):
            return boundsurf(self.l+other, self.u+other, self.definiteRange, self.domain)
        elif other.__class__ == boundsurf:# TODO: replace it by type(r[0]) after dropping Python2 support
            return boundsurf(self.l+other.l, self.u+other.u, self.definiteRange & other.definiteRange, self.domain)
        elif type(other) == np.ndarray:
            assert other.shape[0] == 2, 'unimplemented yet'
            return boundsurf(self.l+other[0], self.u+other[1], self.definiteRange, self.domain)
        else:
            assert 0, 'unimplemented yet'
            
    __radd__ = lambda self, other: self.__add__(other)
    
    def __neg__(self):
        l, u = self.l, self.u
        L = surf(dict([(k, -v) for k, v in u.d.items()]), -u.c)
        U = surf(dict([(k, -v) for k, v in l.d.items()]), -l.c)
        return boundsurf(L, U, self.definiteRange, self.domain)
    
    # TODO: mb rework it
    __sub__ = lambda self, other: self + (-other)
        
    def __mul__(self, other):
        if np.isscalar(other) or (type(other) == np.ndarray and other.size == 1):
            if other >= 0:
                return boundsurf(self.l*other, self.u*other, self.definiteRange, self.domain)
            else:
                return boundsurf(self.u*other, self.l*other, self.definiteRange, self.domain)
#        elif other.__class__ == boundsurf:
#            return boundsurf(self.l+other.l, self.u+other.u)
        else:
            assert 0, 'unimplemented yet'
            
    # TODO: rework it if __iadd_, __imul__ etc will be created
    def copy(self):
        return self
    
    def __pow__(self, other):
        assert np.isscalar(other) and other == 2, 'unimplemented yet'
        L, U = self.l, self.u
        
        #L.render()
        
        R0 = self.resolve()[0]#L.resolve(self.domain, GREATER), U.resolve(self.domain, LESS)
        assert R0.shape[0]==2, 'unimplemented yet'
        abs_R0 = np.abs(R0)
        abs_R0.sort(axis=0)
        abs_min, abs_max = abs_R0
        r_l, r_u = R0
        ind_0 = np.where(np.sign(r_l) != np.sign(r_u))[0]
        abs_min[ind_0] = 0.0
        new_u_resolved = abs_max**2
        
        l1, u1 = np.abs(r_l), np.abs(r_u)
        ind = u1 > l1
        #tmp2 = 2 * abs_max
        tmp2 = 2 * abs_min
        Ld, Ud = L.d, U.d
        dep = set(Ld.keys()) | set(Ud.keys()) 
        d_new = {}
        
        for v in dep:
            #t = tmp2 * np.where(ind, Ud.get(v, 0), Ld.get(v, 0))
            t = tmp2 * np.where(ind, Ld.get(v, 0), Ud.get(v, 0))
            #t[ind_0] = 0.0
            d_new[v] = t
            
        L_new = surf(d_new, 0.0)
        _min = L_new.resolve(self.domain, GREATER)
        L_new.c = abs_min**2 - _min

        R = boundsurf(L_new, surf({}, new_u_resolved), 
        self.definiteRange, self.domain)

        return R
        

def boundsurf_mult(b1, b2):
    d1l, d1u, d2l, d2u = b1.l.d, b1.u.d, b2.l.d, b2.u.d
    c1l, c1u, c2l, c2u = b1.l.c, b1.u.c, b2.l.c, b2.u.c
    
    d_l, d_u = {}, {}
    c_l, c_u = 0.0, 0.0
    
    r = boundsurf(surf(d_l, c_l), surf(d_u, c_u), b1.definiteRange & b2.definiteRange)
    return r
    
def boundsurf_abs(b):
    
    r, definiteRange = b.resolve()
    lf, uf = r
    assert lf.ndim <= 1, 'unimplemented yet'
    sz = lf.size
    
    ind_l = lf >= 0#np.where(lf >= 0)[0]
    if np.all(ind_l):
        return b
    
    ind_u = uf <= 0#np.where(uf <= 0)[0]
    if np.all(ind_u):
        return -b
    l_ind, u_ind = np.where(ind_l), np.where(ind_u)
    d_l, c_l, d_u, c_u = b.l.d, b.l.c, b.u.d, b.u.c
    #ind = ~(ind_l | ind_u)

#    Ld = dict((k, f_abs(sz)) for k in set(d_l.keys()) | set(d_u.keys()))
#    Ud = 
    
    ind = np.where(np.sign(lf) != np.sign(uf))

def f_abs(b, l_ind, u_ind, sz, k):
    l =  np.zeros(sz)
    l[l_ind] = b.l.d[k][l_ind]
