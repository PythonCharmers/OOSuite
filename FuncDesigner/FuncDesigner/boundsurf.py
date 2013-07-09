PythonSum = sum
PythonAll = all
import numpy as np
from numpy import all, any, logical_and, logical_not, isscalar, where, inf, logical_or, logical_xor, isnan
from operator import gt as Greater, lt as Less, truediv as td
from FDmisc import update_mul_inf_zero, update_div_zero
import operator

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax
arrZero = np.array(0.0)

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
    
    def exclude(self, oovars):
        L = self.l.exclude(self.domain, oovars, Greater)
        U = self.u.exclude(self.domain, oovars, Less)
        if len(L.d) != 0 or len(U.d) != 0:
            return boundsurf(L, U, self.definiteRange, self.domain)
        else:
            return np.vstack((L.c, U.c))
    
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
        domain = self.domain
        definiteRange = self.definiteRange
        selfPositive = all(R1 >= 0)
        selfNegative = all(R1 <= 0)
        
        isArray = type(other) == np.ndarray
        isBoundSurf = type(other) == boundsurf
        if isBoundSurf:
            definiteRange = logical_and(definiteRange, other.definiteRange)
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
                
#                tmp_l3 = other_ub[ind_z] * l.extract(ind_z)
#                tmp_u3 = other_ub[ind_z] * u.extract(ind_z)
                
                l2, u2 = other_lb[ind_z], other_ub[ind_z]
                l1, u1 = lb1[ind_z], ub1[ind_z]
                Tmp = np.vstack((l1*l2, l1*u2, l2*u1, u1*u2))
                tmp_l3 = surf({}, nanmin(Tmp, axis=0))
                tmp_u3 = surf({}, nanmax(Tmp, axis=0))
                

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
                lb1, ub1 = R1
                lb2, ub2 = R2
                l, u = self.l, self.u
                ind_other_positive, ind_other_negative, ind_z2 = Split(lb2 >= 0, ub2 <= 0)
                ind_positive, ind_negative, ind_z1 = Split(lb1 >= 0, ub1 <= 0)
                inds, lu = [], []

                ind_Z = where(ind_z1)[0]
                if ind_Z.size:
                    inds.append(ind_Z)
                    l2, u2 = lb2[ind_Z], ub2[ind_Z]
                    l1, u1 = lb1[ind_Z], ub1[ind_Z]
                    Tmp = np.vstack((l1*l2, l1*u2, l2*u1, u1*u2))
                    tmp_l3 = surf({}, nanmin(Tmp, axis=0))
                    tmp_u3 = surf({}, nanmax(Tmp, axis=0))
                    b_z = boundsurf(tmp_l3, tmp_u3, definiteRange, domain)
                    lu.append((b_z.l, b_z.u))
                    
                ind_positive_all = logical_and(ind_positive, ind_other_positive)
                ind_Positive = where(ind_positive_all)[0]
                if ind_Positive.size:
                    inds.append(ind_Positive)
                    L = l.extract(ind_Positive) * lb2[ind_Positive]
                    U = u.extract(ind_Positive) * ub2[ind_Positive]
                    lu.append((L, U))
                
                ind_negative_all = logical_and(ind_negative, ind_other_negative)
                ind_Negative =  where(ind_negative_all)[0]
                if ind_Negative.size:
                    inds.append(ind_Negative)
                    U = l.extract(ind_Negative) * lb2[ind_Negative]
                    L = u.extract(ind_Negative) * ub2[ind_Negative]
                    lu.append((L, U))
                    
                ind = logical_and(ind_positive, ind_other_negative)
                Ind = where(ind)[0]
                if Ind.size:
                    inds.append(Ind)
                    L = u.extract(Ind) * lb2[Ind]
                    U = l.extract(Ind) * ub2[Ind]
                    lu.append((L, U))

                ind = logical_and(ind_negative, ind_other_positive)
                Ind = where(ind)[0]
                if Ind.size:
                    inds.append(Ind)
                    L = l.extract(Ind) * ub2[Ind]
                    U = u.extract(Ind) * lb2[Ind]
                    lu.append((L, U))

                ind = logical_and(ind_positive, ind_z2)
                Ind = where(ind)[0]
                if Ind.size:
                    inds.append(Ind)
                    uu = u.extract(Ind)
                    L = uu * lb2[Ind]
                    U = uu * ub2[Ind]
                    lu.append((L, U))
                
                ind = logical_and(ind_negative, ind_z2)
                Ind = where(ind)[0]
                if Ind.size:
                    inds.append(Ind)
                    ll = l.extract(Ind)
                    L = ll * ub2[Ind]
                    U = ll * lb2[Ind]
                    lu.append((L, U))
#
#                    ind = logical_and(ind_z1, ind_other_positive)
#                    Ind = where(ind)[0]
#                    if Ind.size:
#                        print('8')
#                        inds.append(Ind)
#                        L = l.extract(Ind) * ub2[Ind]
#                        U = u.extract(Ind) * ub2[Ind]
#                        lu.append((L, U))
#                    
#                    ind = logical_and(ind_z1, ind_other_negative)
#                    Ind = where(ind)[0]
#                    if Ind.size:
#                        print('9')
#                        inds.append(Ind)
#                        L = u.extract(Ind) * lb2[Ind]
#                        U = l.extract(Ind) * lb2[Ind]
#                        lu.append((L, U))
#                        
                B = [boundsurf(L, U, False, domain) for L, U in lu]
                rr = boundsurf_join(inds, B)
                rr.definiteRange = definiteRange
        elif isBoundSurf:
            if (selfPositive or selfNegative) and (R2Positive or R2Negative):
                r = ((self if selfPositive else -self).log() + (other if R2Positive else -other).log()).exp()
                r.definiteRange = definiteRange
                rr = r if selfPositive == R2Positive else -r
            else:
                Elems = (self, other)
                rr = aux_mul_div_boundsurf(Elems, operator.mul)

#            else:
#                RR = R1*R2 if selfPositive and R2Positive \
#                else (R1*R2)[::-1] if not selfPositive and not R2Positive\
#                else R1[::-1]*R2 if not selfPositive and R2Positive\
#                else R1*R2[::-1] #if selfPositive and not R2Positive
#                new_l_resolved, new_u_resolved = RR
#                
#                l1, u1, l2, u2 = self.l, self.u, other.l, other.u
#                l, u = l1.koeffs_mul(l2), u1.koeffs_mul(u2)
#                l.c = new_l_resolved - l.minimum(domain)
#                u.c = new_u_resolved - u.maximum(domain)
#                rr = (l, u)

#            return R1*other# if nanmax(R2[0])
            #return 0.5 * (R1*other + R2*self)
        else:
            assert 0, 'bug or unimplemented yet'
        R = rr if type(rr) == boundsurf else boundsurf(rr[0], rr[1], definiteRange, domain)

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
            R_Tmp = boundsurf(surf({}, t_min), surf({}, t_max), definiteRange_Tmp, domain)
            R = R_Tmp if all(ind) else boundsurf_join((ind, logical_not(ind)), (R_Tmp, R_tmp))
        return R
    
    __rmul__ = __mul__
    
    def __div__(self, other):
        isBoundSurf = type(other) == boundsurf
        assert isBoundSurf
        
        r = aux_mul_div_boundsurf((self, other), operator.truediv)
        
#        return r 
#        ind_inf_z = logical_or(logical_or(R2[0]==0, R2[1]==0), logical_or(isinf(R1[0]), isinf(R1[1])))
        #(R2[0]==0) | (R2[1]==0) | (isinf(R2[0])) | (isinf(R2[1])) | (isinf(R1[0])) | isinf(R1[1])
        
        rr = r.resolve()[0]
        # nans may be from other computations from a level below, although
        ind_nan = logical_or(isnan(rr[0]), isnan(rr[1]))
        if not any(ind_nan):
            return r
            
        Ind_finite = where(logical_not(ind_nan))[0]
        r_finite = r.extract(Ind_finite)
        ind_nan = where(ind_nan)[0]
        R1 = self.resolve()[0]
        R2 = other.resolve()[0]
        lb1, ub1, lb2, ub2 = R1[0, ind_nan], R1[1, ind_nan], R2[0, ind_nan], R2[1, ind_nan]
        tmp = np.vstack((td(lb1, lb2), td(lb1, ub2), td(ub1, lb2), td(ub1, ub2)))
        R = np.vstack((nanmin(tmp, 0), nanmax(tmp, 0)))
        update_div_zero(lb1, ub1, lb2, ub2, R)
        b = boundsurf(surf({}, R[0]), surf({}, R[1]), False, self.domain)
        r = boundsurf_join((ind_nan, Ind_finite), (b, r_finite))
        definiteRange = logical_and(self.definiteRange, other.definiteRange)
        r.definiteRange = definiteRange
        return r 
    
    __truediv__ = __div__
    
#    __rdiv__ = lambda self, other: other * self ** -1
    
#    __rtruediv__ = __rdiv__

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
    d = dict((k, Join(inds, [s.d.get(k, arrZero) for s in S])) for k in keys)
    c = Join(inds, [s.c for s in S])
    return surf(d, c)

def boundsurf_join(inds, B):
    inds = [(ind if ind.dtype != bool else where(ind)[0]) for ind in inds]
#    B = [b for b in B if b is not None]
    L = surf_join(inds, [b.l for b in B])
    U = surf_join(inds, [b.u for b in B]) #if self.l is not self.u else L
    definiteRange = True \
    if PythonAll(np.array_equiv(True, b.definiteRange) for b in B)\
    else Join(inds, [np.asarray(b.definiteRange) for b in B])
    return boundsurf(L, U, definiteRange, B[0].domain)

#split = lambda condition1, condition2: \
#    (
#    where(condition1)[0], 
#    where(logical_and(condition2, logical_not(condition1)))[0], 
#    where(logical_and(logical_not(condition1), logical_not(condition2)))[0]
#    )

def split(*conditions):
    Rest = np.ones_like(conditions[0]) # dtype bool
    r = []
    for c in conditions:
        tmp = logical_and(c, Rest)
        r.append(where(tmp)[0])
        Rest &= logical_not(c)
    r.append(where(Rest)[0])
    return r

    
Split = lambda condition1, condition2: \
    (
    condition1, 
    logical_and(condition2, logical_not(condition1)), 
    logical_and(logical_not(condition1), logical_not(condition2))
    )

import ooFun

def devided_interval(inp, r, domain, dtype, feasLB = -inf, feasUB = inf):

    lb_ub, definiteRange = inp._interval(domain, dtype, allowBoundSurf = True)
    isBoundSurf = type(lb_ub) == boundsurf
    if not isBoundSurf:
        return ooFun.oofun._interval_(r, domain, dtype)
    
    lb_ub_resolved = lb_ub.resolve()[0]
    
    if feasLB != -inf or feasUB != inf:
        from Interval import adjustBounds
        lb_ub_resolved, definiteRange = adjustBounds(lb_ub_resolved, definiteRange, feasLB, feasUB)
        lb_ub.definiteRange = definiteRange
        
    lb, ub = lb_ub_resolved
    Inds = split(ub <= -0.0, lb >= 0.0)
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
        if convexities == (-1, 1) and r.engine_monotonity == 1:
            tmp = defaultIntervalEngine(lb_ub, r.fun, r.d, monotonity = r.engine_monotonity, convexity=-101, 
                                        feasLB = feasLB, feasUB = feasUB, domain_ind = _ind)[0]
            if _ind.size == m:
                return tmp, tmp.definiteRange
        elif convexities == (1, -1) and r.engine_monotonity is not np.nan:
            tmp = defaultIntervalEngine(lb_ub, r.fun, r.d, monotonity = r.engine_monotonity, convexity= 9, # 10-1 
                                        feasLB = feasLB, feasUB = feasUB, domain_ind = _ind)[0]
            if _ind.size == m:
                return tmp, tmp.definiteRange
        else:
            DefiniteRange = definiteRange if type(definiteRange) == bool or definiteRange.size == 1 \
            else definiteRange[_ind]
            
            Tmp, definiteRange3 = \
            ooFun.oofun._interval_(r, domain, dtype, inputData = (lb_ub_resolved[:, _ind], DefiniteRange))
            
            if _ind.size == m:
                return Tmp, definiteRange3
            tmp = boundsurf(surf({}, Tmp[0]), surf({}, Tmp[1]), definiteRange3, domain)
            
        rr.append(tmp)
        inds.append(_ind)

    b = boundsurf_join(inds, rr)
    return b, b.definiteRange


def aux_mul_div_boundsurf(Elems, op):
    _r = []
    _resolved = []
    changeSign = False
    indZ = False
    
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
    if op == operator.mul:
        rr = PythonSum(_r).exp()
    else:
        assert op == operator.truediv and len(Elems) == 2
        rr = (_r[0] - _r[1]).exp()
        
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
        lb1, ub1 = Elems[0].resolve()[0] # R1
        other_lb, other_ub = Elems[1].resolve()[0] # R2
        
        IndZ = where(indZ)[0]
        tmp_z = np.vstack((
                           op(lb1[IndZ], other_lb[IndZ]), 
                           op(ub1[IndZ], other_lb[IndZ]), 
                           op(lb1[IndZ], other_ub[IndZ]), 
                           op(ub1[IndZ], other_ub[IndZ])
                           ))
        l_z, u_z = nanmin(tmp_z, 0), nanmax(tmp_z, 0)
        rr_z = boundsurf(surf({}, l_z), surf({}, u_z), True, Elems[0].domain)
        _rr.append(rr_z)
        _inds.append(indZ)
    rr = boundsurf_join(_inds, _rr)
    rr.definiteRange = definiteRange
    return rr




