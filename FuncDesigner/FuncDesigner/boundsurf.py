PythonSum = sum
PythonAll = all
import numpy as np
from numpy import all, any, logical_and, logical_not, logical_or, isscalar, where, inf
from operator import gt as Greater, lt as Less

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax

#def extract(b, ind):
#    d = dict((k, v if v.size == 1 else v[ind]) for k, v in b.d.items())
#    C = b.c
#    c = C if C.size == 1 else C[ind]
#    return surf(d, c)

class surf(object):
    isRendered = False
    __array_priority__ = 15
    def __init__(self, d, c):
        self.d = d # dict of variables and linear coefficients on them (probably as multiarrays)
        self.c = np.asarray(c) # (multiarray of) constant(s)

    value = lambda self, point: self.c + PythonSum(point[k] * v for k, v in self.d.items())

#    resolve = lambda self, domain, cmp: \
#    self.c + PythonSum(where(cmp(v, 0), domain[k][0], domain[k][1])*v for k, v in self.d.items())
    
    def exclude(self, domain, oovars, cmp):
        C = []
        d = self.d.copy()
        for v in oovars:
            tmp = d.pop(v, 0.0)
            if any(tmp):
                D = domain[v]
                C.append(where(cmp(tmp, 0), D[0], D[1])*tmp)
        c = self.c + PythonSum(C)
        return surf(d, c)
    
#    split = lambda self, inds: [extract(self, ind) for ind in inds]
    
    #self.resolve(domain, GREATER)
    minimum = lambda self, domain, domain_ind = slice(None): \
    self.c +\
    PythonSum(where(v > 0, 
    domain[k][0][domain_ind], domain[k][1][domain_ind])*v for k, v in self.d.items())
    
    #self.resolve(domain, LESS)
    maximum = lambda self, domain, domain_ind = slice(None): \
    self.c +\
    PythonSum(where(v  < 0, 
    domain[k][0][domain_ind], domain[k][1][domain_ind])*v for k, v in self.d.items())
    
    def render(self, domain, cmp):
        self.rendered = dict((k, where(cmp(v, 0), domain[k][0], domain[k][1])*v) for k, v in self.d.items())
        self.resolved = PythonSum(self.rendered) + self.c
        self.isRendered = True
    
    def __add__(self, other):
        if type(other) == surf:
#            if other.isRendered and not self.isRendered:
#                self, other = other, self
            S, O = self.d, other.d
            d = S.copy()
            d.update(O)
            for key in set(S.keys()) & set(O.keys()):
                d[key] = S[key]  + O[key]
            return surf(d, self.c+other.c)
        elif isscalar(other) or type(other) == np.ndarray:
            return surf(self.d, self.c + other)
        else:
            assert 0, 'unimplemented yet'
    
    __sub__ = lambda self, other: self.__add__(-other)
    
    __neg__ = lambda self: surf(dict((k, -v) for k, v in self.d.items()), -self.c)
    
    def __mul__(self, other):
        isArray = type(other) == np.ndarray
        if isscalar(other) or isArray:
            return surf(dict((k, v*other) for k, v in self.d.items()), self.c * other)
        else:
            assert 0, 'unimplemented yet'
            
    __rmul__ = __mul__
    
    def koeffs_mul(self, other):
        assert type(other) == surf
        S, O = self.d, other.d
        d = dict((key, S.get(key, 0.0)  * O.get(key, 0.0)) for key in set(S.keys()) | set(O.keys()))
        return surf(d, 0.0)
            
#    def __getattr__(self, attr):
#        if attr == 'resolve_index':
#            assert 0, 'resolve_index must be used from surf derived classes only'
#        else:
#            raise AttributeError('error in FD engine (class surf)')
            

class boundsurf(object):#object is added for Python2 compatibility
    __array_priority__ = 15
    isRendered = False
    def __init__(self, lowersurf, uppersurf, definiteRange, domain):
        self.l = lowersurf
        self.u = uppersurf
        self.definiteRange = definiteRange
        self.domain = domain
        
    Size = lambda self: max((len(self.l.d), len(self.u.d), 1))
    
    exclude = lambda self, oovars:\
        boundsurf(self.l.exclude(self.domain, oovars, Greater), self.u.exclude(self.domain, oovars, Less), 
                  self.definiteRange, self.domain)
    
#    def split(self, condition1, condition2):
#        inds = (
#                where(condition1)[0], 
#                where(logical_and(condition2, logical_not(condition1)))[0], 
#                where(logical_and(logical_not(condition1), logical_not(condition2)))[0]
#                )
#
#        L = self.l.split(inds)
#        U = self.u.split(inds) if self.l is not self.u else L
#        
#        definiteRange = self.definiteRange
#        DefiniteRange = [definiteRange] * len(inds) if type(definiteRange) == bool or definiteRange.size == 1 \
#        else [definiteRange[ind] for ind in inds]
#        
#        return inds, [boundsurf(L[i], U[i], DefiniteRange[i], self.domain) if ind.size else None for i, ind in enumerate(inds)]
    
#    def resolve(self, ind = slice(None)):
#        r = np.vstack((self.l.minimum(self.domain, ind), self.u.maximum(self.domain, ind)))
#        assert r.shape[0] == 2, 'bug in FD kernel'
#        return r, (self.definiteRange if type(self.definiteRange) == bool or self.definiteRange.size == 1 \
#                   else self.definiteRange[ind])
    def resolve(self):
        r = np.vstack((self.l.minimum(self.domain), self.u.maximum(self.domain)))
        assert r.shape[0] == 2, 'bug in FD kernel'
        return r, self.definiteRange
    
    def render(self):
        if self.isRendered:
            return
#        self.l.render(self, self.domain, GREATER)
#        self.u.render(self, self.domain, LESS)
        self.isRendered = True
    
    values = lambda self, point: (self.l.value(point), self.u.value(point))
    
    isfinite = lambda self: all(np.isfinite(self.l.c)) and all(np.isfinite(self.u.c))
    
    # TODO: handling fd.sum()
    def __add__(self, other):
        if isscalar(other) or (type(other) == np.ndarray and other.size == 1):
            if self.l is self.u:
                # TODO: mb use id() instead of "is"
                tmp = self.l+other
                rr = (tmp, tmp)
            else:
                rr = (self.l+other, self.u+other)
            return boundsurf(rr[0], rr[1], self.definiteRange, self.domain)
        elif type(other) == boundsurf:# TODO: replace it by type(r[0]) after dropping Python2 support
            if self.l is self.u and other.l is other.u:
                # TODO: mb use id() instead of "is"
                tmp = self.l+other.l
                rr = (tmp, tmp)
            else:
                rr = (self.l+other.l, self.u+other.u)
            return boundsurf(rr[0], rr[1], self.definiteRange & other.definiteRange, self.domain)
        elif type(other) == np.ndarray:
            assert other.shape[0] == 2, 'unimplemented yet'
            return boundsurf(self.l+other[0], self.u+other[1], self.definiteRange, self.domain)
        else:
            assert 0, 'unimplemented yet'
            
    __radd__ = __add__
    
    def __neg__(self):
        l, u = self.l, self.u
        if l is u:
            tmp = surf(dict((k, -v) for k, v in u.d.items()), -u.c)
            L, U = tmp, tmp
        else: 
            L = surf(dict((k, -v) for k, v in u.d.items()), -u.c)
            U = surf(dict((k, -v) for k, v in l.d.items()), -l.c)
        return boundsurf(L, U, self.definiteRange, self.domain)
    
    # TODO: mb rework it
    __sub__ = lambda self, other: self.__add__(-other)
        
    def __mul__(self, other):
        R1 = self.resolve()[0]
        definiteRange = self.definiteRange
        selfPositive = all(R1 >= 0)
        selfNegative = all(R1 <= 0)
        
        isArray = type(other) == np.ndarray
        isBoundSurf = type(other) == boundsurf
        R2 = other.resolve()[0] if isBoundSurf else other
        R2_is_scalar = isscalar(R2)
        
        if not R2_is_scalar and R2.size != 1:
            assert R2.shape[0] == 2, 'bug or unimplemented yet'
            R2Positive = all(R2 >= 0)
            R2Negative = all(R2 <= 0)
            assert R2Positive or R2Negative, 'bug or unimplemented yet'
            
        if R2_is_scalar or (isArray and R2.size == 1):
            if self.l is self.u:
                tmp = self.l * R2
                rr = (tmp, tmp)
            else:
                rr = (self.l * R2, self.u * R2) if R2 >= 0 else (self.u * R2, self.l * R2)
        elif isArray:
            
            if 1:
                assert selfPositive or selfNegative, 'unimplemented yet'
            
            else:
                assert R2Positive or R2Negative, 'bug or unimplemented yet'
                lb1, ub1 = R1
                Ind1 = lb1 >= 0
                ind1 = where(Ind1)[0]
                Ind2 = ub1 <= 0
                ind2 = where(Ind2)[0]
                ind3 = where(logical_not(logical_or(Ind1, Ind2)))[0]
                tmp_l, tmp_u = np.empty_like(lb1), np.empty_like(ub1)
                
                lb2, ub2 = R2 if R2Positive else (-R2[1], -R2[0])

                tmp_l1 = lb2[ind1] * self.l[ind1]
                tmp_l2 = ub2[ind2] * self.l[ind2]
                l2, u2 = lb2[ind3], ub2[ind3]
                l1, u1 = lb1[ind3], ub1[ind3]
                Tmp = np.vstack((l1*l2, l1*u2, l2*u1, u1*u2))
                tmp_l3 = nanmin(Tmp, axis=0)
                tmp_u3 = nanmax(Tmp, axis=0)
                
                if R2Negative:
                    # TODO: implement revert
                    pass
                rr = (tmp_l, tmp_u)


            if selfPositive: 
                rr = (self.l * R2[0], self.u * R2[1]) if R2Positive else (self.u * R2[0], self.l * R2[1])
            else:#selfNegative
                assert selfNegative
                rr = (self.u * R2[1], self.l * R2[0]) if R2Negative else (self.l * R2[1], self.u * R2[0])
            
        elif isBoundSurf:
            assert selfPositive or selfNegative, 'bug or unimplemented yet'
            definiteRange = logical_and(definiteRange, other.definiteRange)
            if 1:
                r = ((self if selfPositive else -self).log() + (other if R2Positive else -other).log()).exp()
                r.definiteRange = definiteRange
                return r if selfPositive == R2Positive else -r
            else:
                RR = R1*R2 if selfPositive and R2Positive \
                else (R1*R2)[::-1] if not selfPositive and not R2Positive\
                else R1[::-1]*R2 if not selfPositive and R2Positive\
                else R1*R2[::-1] #if selfPositive and not R2Positive
                new_l_resolved, new_u_resolved = RR
                
                l1, u1, l2, u2 = self.l, self.u, other.l, other.u
                l, u = l1.koeffs_mul(l2), u1.koeffs_mul(u2)
                l.c = new_l_resolved - l.minimum(self.domain)
                u.c = new_u_resolved - u.maximum(self.domain)
                rr = (l, u)

#            return R1*other# if nanmax(R2[0])
            #return 0.5 * (R1*other + R2*self)
        else:
            assert 0, 'bug or unimplemented yet'
        
        R = boundsurf(rr[0], rr[1], definiteRange, self.domain)
        return R
    
    __rmul__ = __mul__
    
    def __div__(self, other):
        R1 = self.resolve()[0]
        definiteRange = self.definiteRange
        selfPositive = all(R1 >= 0)
        selfNegative = all(R1 <= 0)
        
#        isArray = type(other) == np.ndarray
        isBoundSurf = type(other) == boundsurf
        assert isBoundSurf
        R2 = other.resolve()[0] #if isBoundSurf else other
#        R2_is_scalar = isscalar(R2)     
        assert R2.shape[0] == 2, 'bug or unimplemented yet'
        R2Positive = all(R2 >= 0)
        R2Negative = all(R2 <= 0)
        assert (selfPositive or selfNegative) and (R2Positive or R2Negative), 'bug or unimplemented yet'
        definiteRange = logical_and(definiteRange, other.definiteRange)
        r = ((self if selfPositive else -self).log() - (other if R2Positive else -other).log()).exp()
        r.definiteRange = definiteRange
        return r if selfPositive == R2Positive else -r
    
    __truediv__ = __div__

    def log(self):
        from Interval import defaultIntervalEngine
        return defaultIntervalEngine(self, np.log, lambda x: 1.0 / x, 
                     monotonity = 1, convexity = -1, feasLB = 0.0)[0]
    def exp(self):
        from Interval import defaultIntervalEngine
        return defaultIntervalEngine(self, np.exp, np.exp, 
                     monotonity = 1, convexity = 1)[0]

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
        
        R0 = self.resolve()[0]#L.resolve(self.domain, GREATER), U.resolve(self.domain, LESS)
        assert R0.shape[0]==2, 'unimplemented yet'
        
        assert isscalar(other) and other in (-1, 2, 0.5), 'unimplemented yet'
        if other == 0.5:
            from Interval import defaultIntervalEngine
            return defaultIntervalEngine(self, np.sqrt, lambda x: 0.5 / np.sqrt(x), 
                         monotonity = 1, convexity = -1, feasLB = 0.0)[0]        
        elif other == 2:
            from Interval import defaultIntervalEngine
            return defaultIntervalEngine(self, lambda x: x**2, lambda x: 2 * x, 
                         monotonity = 1 if all(R0>=0) else -1 if all(R0<=0) else np.nan, 
                         convexity = 1, 
                         criticalPoint = 0.0, criticalPointValue = 0.0)[0]
        elif other == -1:
            from Interval import defaultIntervalEngine
            return defaultIntervalEngine(self, lambda x: 1.0/x, lambda x: -1.0 / x**2, 
                         monotonity = -1, 
                         convexity = 1 if all(R0>=0) else -1 if all(R0<=0) else np.nan, 
                         criticalPoint = np.nan, criticalPointValue = np.nan)[0]        
    
def boundsurf_abs(b):
    r, definiteRange = b.resolve()
    lf, uf = r

    assert lf.ndim <= 1, 'unimplemented yet'
    
    ind_l = lf >= 0
    if all(ind_l):
        return b, b.definiteRange
    
    ind_u = uf <= 0
    if all(ind_u):
        return -b, b.definiteRange
    
    from Interval import defaultIntervalEngine
        
    return defaultIntervalEngine(b, np.abs, np.sign, 
                         monotonity = np.nan, convexity = 1, 
                         criticalPoint = 0.0, criticalPointValue = 0.0)


def Join(inds, arrays):
    r = np.empty(PythonSum(ind.size for ind in inds), arrays[0].dtype)
    for ind, arr in zip(inds, arrays):
        r[ind] = arr
    return r

def surf_join(inds, S):
    keys = set.union(*[set(s.d.keys()) for s in S])
    d = dict((k, Join(inds, [s.d.get(k, 0.0) for s in S])) for k in keys)
    c = Join(inds, [s.c for s in S])
    return surf(d, c)

def boundsurf_join(inds, B):
#    inds = [inds[i] for i, b in enumerate(B) if b is not None]
#    B = [b for b in B if b is not None]
    L = surf_join(inds, [b.l for b in B])
    U = surf_join(inds, [b.u for b in B]) #if self.l is not self.u else L
    definiteRange = True \
    if PythonAll(np.array_equiv(True, b.definiteRange) for b in B)\
    else Join(inds, [b.definiteRange for b in B])
    return boundsurf(L, U, definiteRange, B[0].domain)

split = lambda condition1, condition2: \
    (
    where(condition1)[0], 
    where(logical_and(condition2, logical_not(condition1)))[0], 
    where(logical_and(logical_not(condition1), logical_not(condition2)))[0]
    )

def devided_interval(inp, r, domain, dtype, feasLB = -inf, feasUB = inf):
    lb_ub, definiteRange = inp._interval(domain, dtype, allowBoundSurf = True)
    isBoundSurf = type(lb_ub) == boundsurf
    lb_ub_resolved = lb_ub.resolve()[0] if isBoundSurf else lb_ub
    
    if feasLB != -inf or feasUB != inf:
        from Interval import adjustBounds
        lb_ub_resolved, definiteRange = adjustBounds(lb_ub_resolved, definiteRange, feasLB, feasUB)
        lb_ub.definiteRange = definiteRange
        
    lb, ub = lb_ub_resolved
    Inds = split(ub <= 0, lb >= 0)
    assert len(Inds) == 3
    
    convexities = r.convexities
    monotonities = [r.engine_monotonity] * (len(Inds)-1) if r.engine_monotonity is not np.nan \
    else r.monotonities
    
    m = PythonSum(ind_.size for ind_ in Inds)
    inds, rr = [], []
    
    from Interval import defaultIntervalEngine
    
    for j, ind in enumerate(Inds[:-1]):
        if ind.size != 0:
            tmp = defaultIntervalEngine(lb_ub, r.fun, r.d, monotonity=monotonities[j], 
                                        convexity=convexities[j], feasLB = feasLB, feasUB = feasUB, domain_ind = ind)[0]
            if ind.size == m:
                return tmp, tmp.definiteRange
            rr.append(tmp)
            inds.append(ind)
    
    _ind = Inds[-1]
    if _ind.size:
        DefiniteRange = definiteRange if type(definiteRange) == bool or definiteRange.size == 1 else definiteRange[_ind]
        from ooFun import oofun
        Tmp, definiteRange3 = oofun._interval_(r, domain, dtype, inputData = (lb_ub_resolved[:, _ind], DefiniteRange))
        if _ind.size == m:
            return Tmp, definiteRange3
        tmp = boundsurf(surf({}, Tmp[0]), surf({}, Tmp[1]), definiteRange3, domain)
        rr.append(tmp)
        inds.append(_ind)

    b = boundsurf_join(inds, rr)
    return b, b.definiteRange







