PythonSum = sum
PythonAll = all
import numpy as np
from numpy import all, any, logical_and, logical_not, isscalar, where, inf, logical_or, logical_xor
from operator import gt as Greater, lt as Less
from FDmisc import update_mul_inf_zero, update_negative_int_pow_inf_zero

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax

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
    
    def extract(self, ind): 
#        if ind.dtype == bool:
#            ind = where(ind)[0]
        d = dict((k, v if v.size == 1 else v[ind]) for k, v in self.d.items()) 
        C = self.c 
        c = C if C.size == 1 else C[ind] 
        return surf(d, c) 
    
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
    
    def extract(self, ind): 
        Ind = ind if ind.dtype != bool else where(ind)[0]
        definiteRange = self.definiteRange if type(self.definiteRange) == bool \
        or self.definiteRange.size == 1 else self.definiteRange[ind]
        return boundsurf(self.l.extract(Ind), self.u.extract(Ind), definiteRange, self.domain)
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
    __rsub__ = lambda self, other: (-self).__add__(other)
        
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
#            if not selfPositive and not selfNegative:
#                assert R2Positive or R2Negative, 'bug or unimplemented yet'
            
        if R2_is_scalar or (isArray and R2.size == 1):
            if self.l is self.u:
                tmp = self.l * R2
                rr = (tmp, tmp)
            else:
                rr = (self.l * R2, self.u * R2) if R2 >= 0 else (self.u * R2, self.l * R2)
        elif isArray:
#            assert R2Positive or R2Negative, 'bug or unimplemented yet'

            if selfPositive and R2Positive:
                rr = (self.l * R2[0], self.u * R2[1]) 
            elif selfPositive and R2Negative:
                rr = (self.u * R2[0], self.l * R2[1])
            elif selfNegative and R2Negative:
                rr = (self.u * R2[1], self.l * R2[0]) 
            elif selfNegative and R2Positive:
                rr = (self.l * R2[1], self.u * R2[0])
            elif R2Positive or R2Negative:
                lb1, ub1 = R1
                ind_positive, ind_negative, ind_z = split(lb1 >= 0, ub1 <= 0)
                other_lb, other_ub = R2 if R2Positive else (-R2[1], -R2[0])
                l, u = self.l, self.u
                
                tmp_l1 = other_lb[ind_positive] * l.extract(ind_positive)
                tmp_l2 = other_ub[ind_negative] * l.extract(ind_negative)
                tmp_u1 = other_ub[ind_positive] * u.extract(ind_positive)
                tmp_u2 = other_lb[ind_negative] * u.extract(ind_negative)
                
                l2, u2 = other_lb[ind_z], other_ub[ind_z]
                l1, u1 = lb1[ind_z], ub1[ind_z]
                Tmp = np.vstack((l1*l2, l1*u2, l2*u1, u1*u2))
                tmp_l3 = surf({}, nanmin(Tmp, axis=0))
                tmp_u3 = surf({}, nanmax(Tmp, axis=0))
#                    tmp_l3 = other_ub[ind_z] * self.l.extract(ind_z)
#                    tmp_u3 = other_ub[ind_z] * self.u.extract(ind_z)
                tmp_l = surf_join((ind_positive, ind_negative, ind_z), (tmp_l1, tmp_l2, tmp_l3))
                tmp_u = surf_join((ind_positive, ind_negative, ind_z), (tmp_u1, tmp_u2, tmp_u3))

                rr = (tmp_l, tmp_u) if R2Positive else (-tmp_u, -tmp_l)
            elif selfPositive or selfNegative:
                l, u = (self.l, self.u) if selfPositive else (-self.u, -self.l)
                lb1, ub1 = R1
                other_lb, other_ub = R2
                ind_other_positive, ind_other_negative, ind_z2 = split(other_lb >= 0, other_ub <= 0)
                
                tmp_l1 = other_lb[ind_other_positive] * l.extract(ind_other_positive)
                tmp_l2 = other_lb[ind_other_negative] * u.extract(ind_other_negative)
                tmp_u1 = other_ub[ind_other_positive] * u.extract(ind_other_positive)
                tmp_u2 = other_ub[ind_other_negative] * l.extract(ind_other_negative)
                if 1:
                    uu = u.extract(ind_z2)
                    tmp_l3 = other_lb[ind_z2] * uu
                    tmp_u3 = other_ub[ind_z2] * uu
                else:
                    l2, u2 = other_lb[ind_z2], other_ub[ind_z2]
                    l1, u1 = lb1[ind_z2], ub1[ind_z2]
                    Tmp = np.vstack((l1*l2, l1*u2, l2*u1, u1*u2))
                    tmp_l3 = surf({}, nanmin(Tmp, axis=0))
                    tmp_u3 = surf({}, nanmax(Tmp, axis=0))
                
                tmp_l = surf_join((ind_other_positive, ind_other_negative, ind_z2), (tmp_l1, tmp_l2, tmp_l3))
                tmp_u = surf_join((ind_other_positive, ind_other_negative, ind_z2), (tmp_u1, tmp_u2, tmp_u3))
                rr = (tmp_l, tmp_u) if selfPositive else (-tmp_u, -tmp_l)
            else:
                # TODO: mb improve it
                rr = self * boundsurf(surf({}, R2[0]),surf({}, R2[1]), self.definiteRange, self.domain) 
        elif isBoundSurf:
            #assert selfPositive or selfNegative, 'bug or unimplemented yet'
            definiteRange = logical_and(definiteRange, other.definiteRange)
            if (selfPositive or selfNegative) and (R2Positive or R2Negative):
                r = ((self if selfPositive else -self).log() + (other if R2Positive else -other).log()).exp()
                r.definiteRange = definiteRange
                rr = r if selfPositive == R2Positive else -r
            else:
                _r = []
                _resolved = []
                changeSign = False
                indZ = False
                Elems = (self, other)
                definiteRange = np.array(True)
                for elem in Elems:
                    _R = elem.resolve()[0]
                    lb, ub = _R
                    ind_positive, ind_negative, ind_z = Split(lb >= 0, ub <= 0)
                    not_ind_negative = logical_not(ind_negative)
                    Ind_negative = where(ind_negative)[0]
                    Not_ind_negative = where(not_ind_negative)[0]
                    changeSign = logical_xor(changeSign, ind_negative)
                    indZ = logical_or(indZ, ind_z)
                    tmp1 = elem.log(domain_ind = Not_ind_negative)
                    tmp2 = (-elem).log(domain_ind = Ind_negative)
                    Tmp = boundsurf_join((not_ind_negative, ind_negative), (tmp1, tmp2))
                    _r.append(Tmp)
                    _resolved.append(_R)
                    definiteRange = logical_and(definiteRange, elem.definiteRange)
                rr = PythonSum(_r).exp()
                changeSign = logical_and(changeSign, logical_not(indZ))
                keepSign = logical_and(logical_not(changeSign), logical_not(indZ))
                _rr, _inds = [], []
                if any(keepSign):
                    _rr.append(rr.extract(keepSign))
                    _inds.append(keepSign)
                if any(changeSign):
                    _rr.append(-rr.extract(changeSign))
                    _inds.append(changeSign)
                if any(indZ):
                    assert len(Elems) == 2, 'unimplemented yet'
                    definiteRange = logical_and(self.definiteRange, other.definiteRange)
                    lb1, ub1 = R1
                    other_lb, other_ub = R2
                    
                    IndZ = where(indZ)[0]
                    tmp_z = np.vstack((lb1[IndZ] * other_lb[IndZ], 
                        ub1[IndZ] * other_lb[IndZ], 
                        lb1[IndZ] * other_ub[IndZ], 
                        ub1[IndZ] * other_ub[IndZ]))
                    l_z, u_z = nanmin(tmp_z, 0), nanmax(tmp_z, 0)
                    definiteRange2 = definiteRange if definiteRange.size == 1 else definiteRange[IndZ]
                    rr_z = boundsurf(surf({}, l_z), surf({}, u_z), definiteRange2, self.domain)
                    _rr.append(rr_z)
                    _inds.append(indZ)
                rr = boundsurf_join(_inds, _rr)

#            else:
#                RR = R1*R2 if selfPositive and R2Positive \
#                else (R1*R2)[::-1] if not selfPositive and not R2Positive\
#                else R1[::-1]*R2 if not selfPositive and R2Positive\
#                else R1*R2[::-1] #if selfPositive and not R2Positive
#                new_l_resolved, new_u_resolved = RR
#                
#                l1, u1, l2, u2 = self.l, self.u, other.l, other.u
#                l, u = l1.koeffs_mul(l2), u1.koeffs_mul(u2)
#                l.c = new_l_resolved - l.minimum(self.domain)
#                u.c = new_u_resolved - u.maximum(self.domain)
#                rr = (l, u)

#            return R1*other# if nanmax(R2[0])
            #return 0.5 * (R1*other + R2*self)
        else:
            assert 0, 'bug or unimplemented yet'
        R = rr if type(rr) == boundsurf else boundsurf(rr[0], rr[1], definiteRange, self.domain)

        lb1, ub1 = R1
        lb2, ub2 = (R2, R2) if R2_is_scalar or R2.size == 1 else R2
        ind_z1 = logical_or(lb1 == 0, ub1 == 0)
        ind_z2 = logical_or(lb2 == 0, ub2 == 0)
        ind_i1 = logical_or(np.isinf(lb1), np.isinf(ub1))
        ind_i2 = logical_or(np.isinf(lb2), np.isinf(ub2))
        ind = logical_or(logical_and(ind_z1, ind_i2), logical_and(ind_z2, ind_i1))
        if any(ind):
            R_tmp = R.extract(logical_not(ind))
            t = np.vstack((lb1 * lb2, ub1 * lb2, lb1 * ub2, ub1 * ub2))
            t_min, t_max = np.atleast_1d(nanmin(t, 0)), np.atleast_1d(nanmax(t, 0))
            update_mul_inf_zero(R1, R2, np.vstack((t_min, t_max)))
            definiteRange_Tmp = \
            R.definiteRange if type(R.definiteRange) == bool or R.definiteRange.size == 1\
            else R.definiteRange[ind]
            R_Tmp = boundsurf(surf({}, t_min), surf({}, t_max), definiteRange_Tmp, self.domain)
            R = R_Tmp if all(ind) else boundsurf_join((ind, logical_not(ind)), (R_Tmp, R_tmp))
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
    
    __rdiv__ = lambda self, other: other * self ** -1

    def log(self, domain_ind = slice(None)):
        from Interval import defaultIntervalEngine
        return defaultIntervalEngine(self, np.log, lambda x: 1.0 / x, 
                     monotonity = 1, convexity = -1, feasLB = 0.0, domain_ind = domain_ind)[0]
    def exp(self, domain_ind = slice(None)):
        from Interval import defaultIntervalEngine
        return defaultIntervalEngine(self, np.exp, np.exp, 
                     monotonity = 1, convexity = 1, domain_ind = domain_ind)[0]

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
        
#        from Interval import pow_const_interval
#        return pow_const_interval(self, r, other, domain, dtype)
        
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
            r = defaultIntervalEngine(self, lambda x: 1.0/x, lambda x: -1.0 / x**2, 
                         monotonity = -1, 
                         convexity = 1 if all(R0>=0) else -1 if all(R0<=0) else np.nan, 
                         criticalPoint = np.nan, criticalPointValue = np.nan)[0]    
            ind = logical_or(R0[0] == 0, R0[1] == 0) 
            if any(ind):
                R_tmp = r.extract(logical_not(ind))
                t = 1.0 / R0[:, ind]
                t.sort(axis=0)
                t_min, t_max = t
                update_negative_int_pow_inf_zero(R0[0], R0[1], t_min, t_max, 1.0)
                definiteRange_Tmp = \
                r.definiteRange if type(r.definiteRange) == bool or r.definiteRange.size == 1\
                else r.definiteRange[ind]
                R_Tmp = boundsurf(surf({}, t_min), surf({}, t_max), definiteRange_Tmp, self.domain)
                r = R_Tmp if all(ind) else boundsurf_join((ind, logical_not(ind)), (R_Tmp, R_tmp))
            return r
    
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
    inds = [(ind if ind.dtype != bool else where(ind)[0]) for ind in inds]
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

Split = lambda condition1, condition2: \
    (
    condition1, 
    logical_and(condition2, logical_not(condition1)), 
    logical_and(logical_not(condition1), logical_not(condition2))
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
    
    monotonities = [r.engine_monotonity] * (len(Inds)-1) if r.engine_monotonity is not np.nan \
    else r.monotonities
    
    convexities = [r.engine_convexity] * (len(Inds)-1) if r.engine_convexity is not np.nan else r.convexities
    
    m = PythonSum(ind_.size for ind_ in Inds)
    inds, rr = [], []
    
    from Interval import defaultIntervalEngine
    
    for j, ind in enumerate(Inds[:-1]):
        if ind.size != 0:
            tmp = defaultIntervalEngine(lb_ub, r.fun, r.d, monotonity=monotonities[j], convexity=convexities[j], 
                                        feasLB = feasLB, feasUB = feasUB, domain_ind = ind)[0]
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







