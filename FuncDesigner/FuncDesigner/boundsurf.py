PythonSum = sum
import numpy as np
from operator import le as LESS,  gt as GREATER

class surf:
    isRendered = False
    def __init__(self, d, c):
        self.d = d # dict of variables and linear coefficients on them (probably as multiarrays)
        self.c = c # (multiarray of) constant(s)
        #self.Type = Type # False for lower, True for upper

    def value(self, point):
        return self.c + PythonSum(point[k]*v for k, v in self.d.items())

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
            
    __radd__ = __add__
    
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
    
    __rmul__ = __mul__
    
    # TODO: rework it if __iadd_, __imul__ etc will be created
    def copy(self):
        assert '__iadd__' not in self.__dict__
        assert '__imul__' not in self.__dict__
        assert '__idiv__' not in self.__dict__
        assert '__isub__' not in self.__dict__
        return self
    
    abs = lambda self: boundsurf_abs(self)
    
    def __pow__(self, other):
        # TODO: rework it
        assert np.isscalar(other) and other in (-1, 2, 0.5), 'unimplemented yet'
        if other == 0.5:
            return boundsurf_sqrt(self)
        
        L, U = self.l, self.u
        
        R0 = self.resolve()[0]#L.resolve(self.domain, GREATER), U.resolve(self.domain, LESS)
        assert R0.shape[0]==2, 'unimplemented yet'
        r_l, r_u = R0

        
        if other == 2:
            abs_R0 = np.abs(R0)
            abs_R0.sort(axis=0)
            abs_min, abs_max = abs_R0
            ind_0 = np.where(np.sign(r_l) != np.sign(r_u))[0]
            abs_min[ind_0] = 0.0
            new_u_resolved = abs_max**2
            
            l1, u1 = np.abs(r_l), np.abs(r_u)
            ind = u1 > l1
            tmp2 = 2 * abs_min
            Ld, Ud = L.d, U.d
            dep = set(Ld.keys()) | set(Ud.keys()) 
            d_new = dict((v, tmp2 * np.where(ind, Ld.get(v, 0), Ud.get(v, 0))) for v in dep)
            L_new = surf(d_new, 0.0)
            _min = L_new.resolve(self.domain, GREATER)
            L_new.c = abs_min**2 - _min

            R = boundsurf(L_new, surf({}, new_u_resolved), self.definiteRange, self.domain)
        else:
            assert other == -1, 'unimplemented yet'
            assert np.all(R0>0), 'bug in FD kernel (unimplemented yet)'
            R2 = 1.0 / R0
            #R2.sort(axis=0)
            #new_l_resolved, new_u_resolved = R2
            new_u_resolved, new_l_resolved = R2 # assuming R >= 0
            
            tmp2 = -1.0 / r_u ** 2
            Ld = L.d
            d_new = dict((v, tmp2 * Ld[v]) for v in Ld)
            L_new = surf(d_new, 0.0)
            _min = L_new.resolve(self.domain, GREATER)
            L_new.c = new_l_resolved - _min

            R = boundsurf(L_new, surf({}, new_u_resolved), self.definiteRange, self.domain)
        return R
        

def boundsurf_mult(b1, b2):
    d1l, d1u, d2l, d2u = b1.l.d, b1.u.d, b2.l.d, b2.u.d
    c1l, c1u, c2l, c2u = b1.l.c, b1.u.c, b2.l.c, b2.u.c
    
    d_l, d_u = {}, {}
    c_l, c_u = 0.0, 0.0
    
    r = boundsurf(surf(d_l, c_l), surf(d_u, c_u), b1.definiteRange & b2.definiteRange)
    return r
    
def boundsurf_abs(b):
#    print(b.resolve())
    r, definiteRange = b.resolve()
    lf, uf = r

    assert lf.ndim <= 1, 'unimplemented yet'
    sz = lf.size
    
    ind_l = lf >= 0#np.where(lf >= 0)[0]
    if np.all(ind_l):
        return b, b.definiteRange
    
    ind_u = uf <= 0#np.where(uf <= 0)[0]
    if np.all(ind_u):
        return -b, b.definiteRange
    l_ind, u_ind = np.where(ind_l)[0], np.where(ind_u)[0]

    d_l, c_l, d_u, c_u = b.l.d, b.l.c, b.u.d, b.u.c
#    ind = ~(ind_l | ind_u)

    Ld = dict((k, f_abs(b, l_ind, u_ind, sz, k)) for k in set(d_l.keys()) | set(d_u.keys()))
    c = np.zeros(sz)
#    c = b.l.c
#    if np.isscalar(c) or c.size == 1:
#        c = np.tile(c, sz)
    l_c = b.l.c
    if np.isscalar(l_c) or l_c.size == 1:
        l_c = np.tile(l_c, sz)
    c[ind_l] = l_c[ind_l]
    u_c = b.u.c
    if np.isscalar(u_c) or u_c.size == 1:
        u_c = np.tile(u_c, sz)
    c[ind_u] = -u_c[ind_u]
    
    M = np.max(np.abs(r), axis = 0)
    R = boundsurf(surf(Ld, c), surf({}, M), b.definiteRange, b.domain)
    
    return R, b.definiteRange
    

def f_abs(b, l_ind, u_ind, sz, k):
    l =  np.zeros(sz)
    if l_ind.size:
        tmp = b.l.d[k]
        l[l_ind] = tmp[l_ind] if type(tmp) == np.ndarray and tmp.size > 1 else tmp
    if u_ind.size:
        tmp = -b.u.d[k]
        l[u_ind] = tmp[u_ind] if type(tmp) == np.ndarray and tmp.size > 1 else tmp
    return l


def boundsurf_sqrt(b):
    L, U = b.l, b.u
    R0, definiteRange = b.resolve()
    assert R0.shape[0]==2, 'unimplemented yet'
    r_l, r_u = R0
    ind_negative = r_l < 0
    
    if np.any(ind_negative):
        r_l[ind_negative] = 0.0
        if type(definiteRange) == bool or definiteRange.shape != r_l.shape:
            definiteRange2 = np.empty(r_l.shape, bool)
            definiteRange2.fill(definiteRange)
            definiteRange = definiteRange2
        definiteRange[ind_negative] = False
    
    new_u_resolved = np.sqrt(r_u)
    new_l_resolved = np.sqrt(r_l)
    
    tmp2 = 0.5 / new_u_resolved
    tmp2[new_u_resolved == 0.0] = 0.0
    Ud = U.d
    d_new = dict((v, tmp2 * Ud[v]) for v in Ud)
    U_new = surf(d_new, 0.0)
    _max = U_new.resolve(b.domain, LESS)
    U_new.c = new_u_resolved - _max

    R = boundsurf(surf({}, new_l_resolved), U_new, definiteRange, b.domain)
    return R

