PythonSum = sum
import numpy as np
from operator import le as LESS,  gt as GREATER

class surf:
    isRendered = False
    def __init__(self, d, c):
        self.d = d # dict of variables and linear coefficients on them (probably as multiarrays)
        self.c = c # (multiarray of) constant(s)
        #self.Type = Type # False for lower, True for upper

    value = lambda self, point: self.c + PythonSum(point[k]*v for k, v in self.d.items())

    resolve = lambda self, domain, cmp: \
    self.c + PythonSum(np.where(cmp(v, 0), domain[k][0], domain[k][1])*v for k, v in self.d.items())
    
    minimum = lambda self, domain: self.resolve(domain, GREATER)
    maximum = lambda self, domain: self.resolve(domain, LESS)
    
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
            
    __rmul__ = __mul__
            
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
        
    Size = lambda self: max((len(self.l.d), len(self.u.d), 1))
        
    def resolve(self):
        l = self.l.resolve(self.domain, GREATER)
        u = self.u.resolve(self.domain, LESS)
        r = np.vstack((l, u))
        assert r.shape[0] == 2, 'bug in FD kernel'
        return r, self.definiteRange
    
    def render(self):
        if self.isRendered:
            return
        self.l.render(self, self.domain, GREATER)
        self.u.render(self, self.domain, LESS)
        self.isRendered = True
    
    values = lambda self, point: (self.l.value(point), self.u.value(point))
    
    isfinite = lambda self: np.all(np.isfinite(self.l.c)) and np.all(np.isfinite(self.u.c))
    
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
    __sub__ = lambda self, other: self.__add__(-other)
        
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
        
        L, U, domain = self.l, self.u, self.domain
        
        R0 = self.resolve()[0]#L.resolve(self.domain, GREATER), U.resolve(self.domain, LESS)
        assert R0.shape[0]==2, 'unimplemented yet'
        lb, ub = R0

        
        if other == 2:
            abs_R0 = np.abs(R0)
            abs_R0.sort(axis=0)
            abs_min, abs_max = abs_R0
            ind_0 = np.where(np.sign(lb) != np.sign(ub))[0]
            abs_min[ind_0] = 0.0
            new_u_resolved = abs_max**2
            
            l1, u1 = np.abs(lb), np.abs(ub)
            ind = u1 > l1
            tmp2 = 2 * abs_min
            Ld, Ud = L.d, U.d
            dep = set(Ld.keys()) | set(Ud.keys()) 
            d_new = dict((v, tmp2 * np.where(ind, Ld.get(v, 0), Ud.get(v, 0))) for v in dep)
            L_new = surf(d_new, 0.0)
            _val = L_new.minimum(domain)
            L_new.c = abs_min**2 - _val

            if 1 and len(Ud) == 1:# and np.all(lb != ub):
                koeffs = ub + lb #(ub^2 - lb^2) / (ub - lb)
                d_new = dict((v, koeffs * val) for v, val in Ud.items())
                U_new = surf(d_new, 0.0)
                _val = U_new.maximum(domain)
                U_new.c = new_u_resolved - _val
            else:
                U_new = surf({}, new_u_resolved)

            R = boundsurf(L_new, U_new, self.definiteRange, domain)
        else:
            assert other == -1, 'unimplemented yet'
            assert np.all(R0>0), 'bug in FD kernel (unimplemented yet)'
            R2 = 1.0 / R0
            #R2.sort(axis=0)
            #new_l_resolved, new_u_resolved = R2
            new_u_resolved, new_l_resolved = R2 # assuming R >= 0
            
            tmp2 = -1.0 / ub ** 2
            Ld, Ud = L.d, U.d
            d_new = dict((v, tmp2 * val) for v, val in Ud.items())
            L_new = surf(d_new, 0.0)
            _val = L_new.minimum(domain)
            L_new.c = new_l_resolved - _val

            if 1 and len(Ud) == 1:# and np.all(lb != ub):
                koeffs = -1.0 /(ub*lb) #(1/ub - 1/lb) / (ub - lb)
                d_new = dict((v, koeffs * val) for v, val in Ld.items())
                U_new = surf(d_new, 0.0)
                _val = U_new.maximum(domain)
                U_new.c = new_u_resolved - _val
            else:
                U_new = surf({}, new_u_resolved)

            R = boundsurf(L_new, U_new, self.definiteRange, domain)
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
    
    ind_l = lf >= 0
    if np.all(ind_l):
        return b, b.definiteRange
    
    ind_u = uf <= 0
    if np.all(ind_u):
        return -b, b.definiteRange
    l_ind, u_ind = np.where(ind_l)[0], np.where(ind_u)[0]

    d_l, c_l, d_u, c_u = b.l.d, b.l.c, b.u.d, b.u.c

    Ld = dict((k, f_abs(b, l_ind, u_ind, sz, k)) for k in set(d_l.keys()) | set(d_u.keys()))
    c = np.zeros(sz)

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
    L, U, domain = b.l, b.u, b.domain
    R0, definiteRange = b.resolve()
    assert R0.shape[0]==2, 'unimplemented yet'
    lb, ub = R0
    ind_negative = lb < 0
    
    if np.any(ind_negative):
        lb[ind_negative] = 0.0
        if type(definiteRange) == bool or definiteRange.shape != lb.shape:
            definiteRange2 = np.empty(lb.shape, bool)
            definiteRange2.fill(definiteRange)
            definiteRange = definiteRange2
        definiteRange[ind_negative] = False
    
    new_u_resolved = np.sqrt(ub)
    new_l_resolved = np.sqrt(lb)
    
    tmp2 = 0.5 / new_u_resolved
    tmp2[new_u_resolved == 0.0] = 0.0
    Ud = U.d
    d_new = dict((v, tmp2 * val) for v, val in Ud.items())
    U_new = surf(d_new, 0.0)
    _val = U_new.maximum(domain)
    U_new.c = new_u_resolved - _val
    
    Ld = L.d
    if 0 and len(Ld) == 1 and np.all(lb != ub):
        koeffs = (new_u_resolved - new_l_resolved) / (ub - lb)
        d_new = dict((v, koeffs * val) for v, val in Ld.items())
        L_new = surf(d_new, 0.0)
        _val = L_new.minimum(domain)
        L_new.c = new_l_resolved - _val
    else:
        L_new = surf({}, new_l_resolved)
    R = boundsurf(L_new, U_new, definiteRange, domain)
    return R

