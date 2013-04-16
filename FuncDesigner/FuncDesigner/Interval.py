from numpy import ndarray, asscalar, isscalar, floor, pi, inf, nan, \
copy as Copy, logical_and, logical_or, where, asarray, any, all, atleast_1d, vstack, \
searchsorted, logical_not
import numpy as np
from FDmisc import FuncDesignerException, Diag
from FuncDesigner.multiarray import multiarray
from boundsurf import boundsurf, surf


try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax

class Interval:
    def __init__(self, l, u, definiteRange):
        if isinstance(l, ndarray) and l.size == 1: l = asscalar(l)
        if isinstance(u, ndarray) and u.size == 1: u = asscalar(u)
        self.lb, self.ub, self.definiteRange = l, u, definiteRange
    def __str__(self):
        return 'FuncDesigner interval with lower bound %s and upper bound %s' % (self.lb, self.ub)
    def __repr__(self):
        return str(self)


def ZeroCriticalPoints(lb_ub):
    arg_infinum, arg_supremum = lb_ub[0], lb_ub[1]
    if isscalar(arg_infinum):
        return [0.0] if arg_infinum < 0.0 < arg_supremum else []
    tmp = Copy(arg_infinum)
    #tmp[where(logical_and(arg_infinum < 0.0, arg_supremum > 0.0))] = 0.0
    tmp[atleast_1d(logical_and(arg_infinum < 0.0, arg_supremum > 0.0))] = 0.0
    return [tmp]

#def IntegerCriticalPoints(arg_infinum, arg_supremum):
#    # TODO: check it for rounding errors
#    return arange(ceil(arg_infinum), ceil(1.0+arg_supremum), dtype=float).tolist()


# TODO: split TrigonometryCriticalPoints into (pi/2) *(2k+1) and (pi/2) *(2k)
def TrigonometryCriticalPoints(lb_ub):
    arg_infinum, arg_supremum = lb_ub[0], lb_ub[1]
    # returns points with coords n * pi/2, arg_infinum <= n * pi/2<= arg_supremum,n -array of integers
    arrN = asarray(atleast_1d(floor(2 * arg_infinum / pi)), int)
    Tmp = []
    for i in range(1, 6):
        th = (arrN+i)*pi/2
        #ind = where(logical_and(arg_infinum < th,  th < arg_supremum))[0]
        ind = logical_and(arg_infinum < th,  th < arg_supremum)
        #if ind.size == 0: break
        if not any(ind): break
        tmp = atleast_1d(Copy(arg_infinum))
        tmp[atleast_1d(ind)] = asarray((arrN[ind]+i)*pi/2, dtype = tmp.dtype)
        Tmp.append(tmp)
    return Tmp
    # 6 instead of  5 for more safety, e.g. small numerical rounding effects
    #return [i / 2.0 * pi for i in range(n1, amin((n1+6, n2))) if (arg_infinum < (i / 2.0) * pi <  arg_supremum)]

#def halph_pi_x_2k_plus_one_points(arg_infinum, arg_supremum):
#    n1 = asarray(floor(2 * arg_infinum / pi), int)
#    Tmp = []
#    for i in range(1, 7):
#        if i% 2: continue
#        ind = where(logical_and(arg_infinum < (n1+i)*pi/2,  (n1+i)*pi/2< arg_supremum))[0]
#        if ind.size == 0: break
#        tmp = arg_infinum.copy()
#        #assert (n1+i)*pi/2 < 6.3
#        tmp[ind] = (n1[ind]+i)*pi/2
#        Tmp.append(tmp)
#    #raise 0
#    return Tmp
#    

cosh_deriv = lambda x: Diag(np.sinh(x))
def ZeroCriticalPointsInterval(inp, func):
    is_abs = func == np.abs
    is_cosh = func == np.cosh    
    def interval(domain, dtype):
        allowBoundSurf = is_abs or is_cosh
        lb_ub, definiteRange = inp._interval(domain, dtype, allowBoundSurf = allowBoundSurf)
        if type(lb_ub) == boundsurf:
            if is_abs:
                return lb_ub.abs()
            elif is_cosh:
                return defaultIntervalEngine(lb_ub, func, cosh_deriv, np.nan, 1, 0.0, 1.0)
            else:
               assert 0, 'bug or unimplemented yet' 
        
        lb, ub = lb_ub[0], lb_ub[1]
        ind1, ind2 = lb < 0.0, ub > 0.0
        ind = logical_and(ind1, ind2)
        tmp = vstack((lb, ub))
        TMP = func(tmp)
        t_min, t_max = atleast_1d(nanmin(TMP, 0)), atleast_1d(nanmax(TMP, 0))
        if any(ind):
            F0 = func(0.0)
            t_min[atleast_1d(logical_and(ind, t_min > F0))] = F0
            t_max[atleast_1d(logical_and(ind, t_max < F0))] = F0
        return vstack((t_min, t_max)), definiteRange
    return interval

def nonnegative_interval(inp, func, domain, dtype, F0, shift = 0.0):

    lb_ub, definiteRange = inp._interval(domain, dtype, allowBoundSurf = True)
    if type(lb_ub) == boundsurf:
        if 1 and func == np.sqrt:
            r = lb_ub ** 0.5
            return r, r.definiteRange
        else:
            lb_ub = lb_ub.resolve()[0]
            
    lb, ub = lb_ub[0], lb_ub[1]
    
    t_min_max = func(lb_ub)
    th = shift # 0.0 + shift = shift
    ind = lb < th
    if any(ind):
        t_min_max[0][atleast_1d(logical_and(lb < th, ub >= th))] = F0
        if definiteRange is not False:
            if type(definiteRange) != np.ndarray:
                definiteRange = np.empty_like(lb)
                definiteRange.fill(True)
            definiteRange[ind] = False
            
    return t_min_max, definiteRange

def box_1_interval(inp, func, domain, dtype, F_l, F_u):
    lb_ub, definiteRange = inp._interval(domain, dtype)
    lb, ub = lb_ub[0], lb_ub[1]
    t_min_max = func(lb_ub)
    
    ind = lb < -1
    if any(ind):
        t_min_max[0][atleast_1d(logical_and(lb < -1, ub >= -1))] = F_l
        if definiteRange is not False:
            if definiteRange is True:
                definiteRange = np.empty_like(lb)
                definiteRange.fill(True)
            definiteRange[ind] = False
        
    ind = ub > 1
    if any(ind):
        t_min_max[0][atleast_1d(logical_and(lb < 1, ub >= 1))] = F_u
        if definiteRange is not False:
            if definiteRange is True:
                definiteRange = np.empty_like(lb)
                definiteRange.fill(True)
            definiteRange[ind] = False
        
    return t_min_max, definiteRange
        
#    lb, ub = lb_ub[0], lb_ub[1]
#    ind = lb < -1.0
#    if any(ind):
#        t_min, t_max = atleast_1d(empty_like(lb)), atleast_1d(empty_like(ub))
#        t_min.fill(nan)
#        t_max.fill(nan)
#        ind2 = ub >= -1.0
#        t_min[atleast_1d(logical_not(ind))] = func(lb)
#        t_min[atleast_1d(logical_and(ind2, ind))] = 0.0
#        t_max[atleast_1d(ind2)] = func(ub[ind2])
#        t_min, t_max = func(lb), func(ub)
#        t_min[atleast_1d(logical_and(lb < 0, ub > 0))] = 0.0
#        
#        # TODO: rework it with matrix operations
#        definiteRange = False
#        
#    else:
#        t_min, t_max = func(lb), func(ub)
#    return vstack((t_min, t_max)), definiteRange

def adjust_lx_WithDiscreteDomain(Lx, v):
    if v.domain is bool or v.domain is 'bool':
        Lx[Lx != 0] = 1
    else:
        d = v.domain 
        ind = searchsorted(d, Lx, 'left')
        ind2 = searchsorted(d, Lx, 'right')
        ind3 = where(ind!=ind2)[0]
        #Tmp = Lx[:, ind3].copy()
        Tmp = d[ind[ind3]]
        #if any(ind==d.size):print 'asdf'
        ind[ind==d.size] -= 1# Is it ever encountered?
    #    ind[ind==d.size-1] -= 1
        Lx[:] = d[ind]
        Lx[ind3] = asarray(Tmp, dtype=Lx.dtype)

        
def adjust_ux_WithDiscreteDomain(Ux, v):
    if v.domain is bool or v.domain is 'bool':
        Ux[Ux != 1] = 0
    else:
        d = v.domain 
        ind = searchsorted(d, Ux, 'left')
        ind2 = searchsorted(d, Ux, 'right')
        ind3 = where(ind!=ind2)[0]
        #Tmp = Ux[:, ind3].copy()
        Tmp = d[ind[ind3]]
        #ind[ind==d.size] -= 1
        ind[ind==0] = 1
        Ux[:] = d[ind-1]
        Ux[ind3] = asarray(Tmp, dtype=Ux.dtype)

def add_interval(self, other, domain, dtype):
    domain1, definiteRange1 = self._interval(domain, dtype, allowBoundSurf = True)
    domain2, definiteRange2 = other._interval(domain, dtype, allowBoundSurf = True)
    return domain1 + domain2, logical_and(definiteRange1, definiteRange2)

def add_const_interval(self, c, domain, dtype): 
    r, definiteRange = self._interval(domain, dtype, allowBoundSurf = True)
    return r + c, definiteRange

def neg_interval(self, domain, dtype):
    r, definiteRange = self._interval(domain, dtype, allowBoundSurf=True)
    if type(r) == ndarray:
        assert r.shape[0] == 2
        #return (-r[1], -r[0])
        return -np.flipud(r), definiteRange
    else:
        #assert type(r) == boundsurf
        return -r, definiteRange

def mul_interval(self, other, isOtherOOFun, domain, dtype):#*args, **kw):
#    if domain.isMultiPoint and isOtherOOFun and self.is_oovar and (self.domain is bool or self.domain is 'bool'):
#        # TODO: add handling allowBoundSurf here
#        lb_ub, definiteRange = other._interval(domain, dtype)
#        n = domain[self][1].size
#        R = np.zeros((2, n), dtype)
#        ind = domain[self][0]!=0
#        R[0][ind] = lb_ub[0][ind]
#        ind = domain[self][1]!=0
#        R[1][ind] = lb_ub[1][ind]
#        return R, definiteRange
    
    lb1_ub1, definiteRange = self._interval(domain, dtype, allowBoundSurf = True)

    if isOtherOOFun:
        lb2_ub2, definiteRange2 = other._interval(domain, dtype, allowBoundSurf = True)
        definiteRange = logical_and(definiteRange, definiteRange2)
        
        if type(lb2_ub2) != boundsurf and type(lb1_ub1) == boundsurf:
            lb2_ub2, lb1_ub1 = lb1_ub1, lb2_ub2
        tmp1 = lb1_ub1.resolve()[0] if type(lb1_ub1) == boundsurf else lb1_ub1
        
        if type(lb2_ub2) == boundsurf:
            tmp2 = lb2_ub2.resolve()[0]
            t2_positive = all(tmp2 >= 0)
            t2_negative = all(tmp2 <= 0)
            if type(lb1_ub1) != boundsurf and (t2_positive or t2_negative):
                t1_positive = all(tmp1 >= 0)
                t1_negative = all(tmp1 <= 0)
                if (t1_positive or t1_negative) \
                and not any(logical_and(tmp1==0, np.isinf(tmp2)))\
                and not any(logical_and(tmp2==0, np.isinf(tmp1))):
                    # TODO: improve it
                    r = (lb1_ub1 if t1_positive else -lb1_ub1) * (lb2_ub2 if t2_positive else -lb2_ub2)
                    if t1_positive != t2_positive:
                        r = -r
                    return r, r.definiteRange
            elif domain.surf_preference and all(np.isfinite(tmp1)) and all(np.isfinite(tmp2)):
                r = 0.25 * ((lb1_ub1 + lb2_ub2) ** 2 - (lb1_ub1 - lb2_ub2) ** 2)
                domain.exactRange = False
                return r, r.definiteRange
#                        
##                    #rr = 0.5*(lb1_ub1 * tmp2 + lb2_ub2 * tmp1)
##                    from ooPoint import ooPoint as oopoint
##                    centers = oopoint((v, asarray(0.5*(val[0] + val[1]))) for v, val in domain.items())
##                    tmp1 = np.linalg.norm(rr.values(centers)[0] - rr.values(centers)[1]) 
##                    tmp2 = np.linalg.norm(r.values(centers)[0] - r.values(centers)[1])
##                    print(tmp1-tmp2, tmp1, tmp2)
##                    tmp1 = np.linalg.norm(rr.resolve()[0][0] - rr.resolve()[0][1]) 
##                    tmp2 = np.linalg.norm(r.resolve()[0][0] - r.resolve()[0][1])
##                    print(tmp1-tmp2, tmp1, tmp2)
##                    print('------')
#
##                    r.definiteRange = definiteRange
#                    return r, r.definiteRange
        else:
            tmp2 = lb2_ub2
        
        lb2, ub2 = tmp2[0], tmp2[1]
    else:
        if type(lb1_ub1) == boundsurf:# TODO: replace it by type(r[0]) after dropping Python2 support
            assert isscalar(other) or other.size==1, 'bug in FD kernel'
            return lb1_ub1 * other, definiteRange
        lb2, ub2 = other, other # TODO: improve it

    if type(lb1_ub1) == boundsurf:
        lb1_ub1 = lb1_ub1.resolve()[0]
    lb1, ub1 = lb1_ub1[0], lb1_ub1[1]
    
    firstPositive = all(lb1 >= 0)
    firstNegative = all(ub1 <= 0)
    secondPositive = all(lb2 >= 0)
    secondNegative = all(ub2 <= 0)
    if firstPositive and secondPositive:
        t_min, t_max = atleast_1d(lb1 * lb2), atleast_1d(ub1 * ub2)
    elif firstNegative and secondNegative:
        t_min, t_max = atleast_1d(ub1 * ub2), atleast_1d(lb1 * lb2)
    elif firstPositive and secondNegative:
        t_min, t_max = atleast_1d(lb2*ub1), atleast_1d(lb1 * ub2)
    elif firstNegative and secondPositive:
        t_min, t_max = atleast_1d(lb1 * ub2), atleast_1d(lb2*ub1)
    elif isscalar(other):
        t_min, t_max = (lb1 * other, ub1 * other) if other >= 0 else (ub1 * other, lb1 * other)
    else:
        if isOtherOOFun:
            t = vstack((lb1 * lb2, ub1 * lb2, \
                        lb1 * ub2, ub1 * ub2))# TODO: improve it
        else:
            t = vstack((lb1 * other, ub1 * other))# TODO: improve it
        t_min, t_max = atleast_1d(nanmin(t, 0)), atleast_1d(nanmax(t, 0))
    #assert isinstance(t_min, ndarray) and isinstance(t_max, ndarray), 'Please update numpy to more recent version'
    
    if any(np.isinf(lb1)) or any(np.isinf(lb2)) or any(np.isinf(ub1)) or any(np.isinf(ub2)):
        ind1_zero_minus = logical_and(lb1<0, ub1>=0)
        ind1_zero_plus = logical_and(lb1<=0, ub1>0)
        
        ind2_zero_minus = logical_and(lb2<0, ub2>=0)
        ind2_zero_plus = logical_and(lb2<=0, ub2>0)
        
        has_plus_inf_1 = logical_or(logical_and(ind1_zero_minus, lb2==-inf), logical_and(ind1_zero_plus, ub2==inf))
        has_plus_inf_2 = logical_or(logical_and(ind2_zero_minus, lb1==-inf), logical_and(ind2_zero_plus, ub1==inf))
        
        # !!!! lines with zero should be before lines with inf !!!!
        ind = logical_or(logical_and(lb1==-inf, ub2==0), logical_and(lb2==-inf, ub1==0))
        t_max[atleast_1d(logical_and(ind, t_max<0.0))] = 0.0
        
        t_max[atleast_1d(logical_or(has_plus_inf_1, has_plus_inf_2))] = inf
        t_max[atleast_1d(logical_or(logical_and(lb1==0, ub2==inf), logical_and(lb2==0, ub1==inf)))] = inf
        
        has_minus_inf_1 = logical_or(logical_and(ind1_zero_plus, lb2==-inf), logical_and(ind1_zero_minus, ub2==inf))
        has_minus_inf_2 = logical_or(logical_and(ind2_zero_plus, lb1==-inf), logical_and(ind2_zero_minus, ub1==inf))
        # !!!! lines with zero should be before lines with -inf !!!!
        t_min[atleast_1d(logical_or(logical_and(lb1==0, ub2==inf), logical_and(lb2==0, ub1==inf)))] = 0.0
        t_min[atleast_1d(logical_or(logical_and(lb1==-inf, ub2==0), logical_and(lb2==-inf, ub1==0)))] = 0.0
        
        t_min[atleast_1d(logical_or(has_minus_inf_1, has_minus_inf_2))] = -inf
    
#            assert not any(isnan(t_min)) and not any(isnan(t_max))
   
    return vstack((t_min, t_max)), definiteRange


def div_interval(self, other, domain, dtype):
    lb1_ub1, definiteRange1 = self._interval(domain, dtype, allowBoundSurf = True)
    lb2_ub2, definiteRange2 = other._interval(domain, dtype, allowBoundSurf = True)
    
    # TODO: mention in doc definiteRange result for 0 / 0
    definiteRange = logical_and(definiteRange1, definiteRange2)

    tmp1 = lb1_ub1.resolve()[0] if type(lb1_ub1) == boundsurf else lb1_ub1
    t1_positive = all(tmp1 >= 0)
    t1_negative = all(tmp1 <= 0)

    tmp2 = lb2_ub2.resolve()[0] if type(lb2_ub2) == boundsurf else lb2_ub2
    t2_strictly_positive = all(tmp2 > 0)
    t2_strictly_negative = all(tmp2 < 0)
    
    if (type(lb1_ub1) == boundsurf  or type(lb2_ub2) == boundsurf) and \
    (t1_positive or t1_negative) and (t2_strictly_positive or t2_strictly_negative):
        changeSign = t1_positive != t2_strictly_positive
        tmp1 = lb1_ub1 if t1_positive else -lb1_ub1
        if type(lb2_ub2) == boundsurf:
            tmp2 = (lb2_ub2 if t2_strictly_positive else -lb2_ub2) ** (-1)
        else:
            assert tmp2.shape[0] == 2
            tmp2 = 1.0 / tmp2[::-1]
        
        tmp = tmp1 *  tmp2
        return (-tmp if changeSign else tmp), definiteRange
    
    lb1, ub1 = tmp1[0], tmp1[1]
    lb2, ub2 = tmp2[0], tmp2[1]
    lb2, ub2 = asarray(lb2, dtype), asarray(ub2, dtype)

    tmp = vstack((lb1/lb2, lb1/ub2, ub1/lb2, ub1/ub2))
    r1, r2 = nanmin(tmp, 0), nanmax(tmp, 0)
    
    ind = logical_or(lb1==0.0, ub1==0.0)
    r1[atleast_1d(logical_and(ind, r1>0.0))] = 0.0
    r2[atleast_1d(logical_and(ind, r2<0.0))] = 0.0

    # adjust inf
    ind2_zero_minus = logical_and(lb2<0, ub2>=0)
    ind2_zero_plus = logical_and(lb2<=0, ub2>0)
    
    r1[atleast_1d(logical_or(logical_and(ind2_zero_minus, ub1>0), logical_and(ind2_zero_plus, lb1<0)))] = -inf
    r2[atleast_1d(logical_or(logical_and(ind2_zero_minus, lb1<0), logical_and(ind2_zero_plus, ub1>0)))] = inf
    
    #assert not any(isnan(r1)) and not any(isnan(r2))
    #assert all(r1 <= r2)
    return vstack((r1, r2)), definiteRange

def rdiv_interval(self, other, domain, dtype):
    arg_lb_ub, definiteRange = self._interval(domain, dtype, allowBoundSurf = True)
    if type(arg_lb_ub) == boundsurf:
        arg_lb_ub_resolved = arg_lb_ub.resolve()[0]
        if all(arg_lb_ub_resolved > 0):
            return other * arg_lb_ub ** (-1), definiteRange
        else:
            arg_lb_ub = arg_lb_ub_resolved
    arg_infinum, arg_supremum = arg_lb_ub[0], arg_lb_ub[1]
    if other.size != 1: 
        raise FuncDesignerException('this case for interval calculations is unimplemented yet')
    r = vstack((other / arg_supremum, other / arg_infinum))
    r1, r2 = nanmin(r, 0), nanmax(r, 0)
    ind_zero_minus = logical_and(arg_infinum<0, arg_supremum>=0)
    if any(ind_zero_minus):
        r1[atleast_1d(logical_and(ind_zero_minus, other>0))] = -inf
        r2[atleast_1d(logical_and(ind_zero_minus, other<0))] = inf
        
    ind_zero_plus = logical_and(arg_infinum<=0, arg_supremum>0)
    if any(ind_zero_plus):
        r1[atleast_1d(logical_and(ind_zero_plus, other<0))] = -inf
        r2[atleast_1d(logical_and(ind_zero_plus, other>0))] = inf

    return vstack((r1, r2)), definiteRange

def pow_const_interval(self, other, domain, dtype):
    lb_ub, definiteRange = self._interval(domain, dtype, allowBoundSurf = True)
    if type(lb_ub) == boundsurf:
        lb_ub_resolved = lb_ub.resolve()[0]

        if other >= 2 and asarray(other, int) == other:
            if other % 2 == 0 or all(lb_ub_resolved >= 0):
                return defaultIntervalEngine(lb_ub, lambda x: x**other, lambda x: other * x**(other-1), \
                                         np.nan, 1, criticalPoint = 0, criticalPointValue = 0)
            elif all(lb_ub_resolved <= 0):
                return defaultIntervalEngine(lb_ub, lambda x: x**other, lambda x: other * x**(other-1), \
                                         np.nan, -1, criticalPoint = 0, criticalPointValue = 0)

    allowBoundSurf = True if isscalar(other) and other in (2, 0.5) else False
    if other == -1 and all(lb_ub_resolved > 0):
        allowBoundSurf = True
        
    if type(lb_ub) == boundsurf:
        if allowBoundSurf:
            return lb_ub**other, definiteRange
        else:
            lb_ub = lb_ub_resolved
        
    Tmp = lb_ub ** other
    
    t_min, t_max = nanmin(Tmp, 0), nanmax(Tmp, 0)
    lb, ub = lb_ub[0], lb_ub[1]
    ind = lb < 0.0
    if any(ind):
        isNonInteger = other != asarray(other, int) # TODO: rational numbers?
        
        # TODO: rework it properly, with matrix operations
        if any(isNonInteger):
            definiteRange = False
        
        ind_nan = logical_and(logical_and(ind, isNonInteger), ub < 0)
        if any(ind_nan):
            t_max[atleast_1d(ind_nan)] = nan
        
        #1
        t_min[atleast_1d(logical_and(ind, logical_and(t_min>0, ub >= 0)))] = 0.0
        
#                    #2
#                    if asarray(other).size == 1:
#                        IND = not isNonInteger
#                    else:
#                        ind2 = logical_not(isNonInteger)
#                        IND = other[ind2] % 2 == 0
#                    
#                    if any(IND):
#                        t_min[logical_and(IND, atleast_1d(logical_and(lb<0, ub >= 0)))] = 0.0
    return vstack((t_min, t_max)), definiteRange
    
def pow_oofun_interval(self, other, domain, dtype): 
    # TODO: handle discrete cases
    lb1_ub1, definiteRange1 = self._interval(domain, dtype)
    lb1, ub1 = lb1_ub1[0], lb1_ub1[1]
    lb2_ub2, definiteRange2 = other._interval(domain, dtype)
    lb2, ub2 = lb2_ub2[0], lb2_ub2[1]
    T = vstack((lb1 ** lb2, lb1** ub2, ub1**lb1, ub1**ub2))
    t_min, t_max = nanmin(T, 0), nanmax(T, 0)
    definiteRange = logical_and(definiteRange1, definiteRange2)
    ind1 = lb1 < 0
    # TODO: check it, especially with integer "other"
    if any(ind1):
        
        # TODO: rework it with matrix operations
        definiteRange = False
        
        ind2 = ub1 >= 0
        t_min[atleast_1d(logical_and(logical_and(ind1, ind2), logical_and(t_min > 0.0, ub2 > 0.0)))] = 0.0
        t_max[atleast_1d(logical_and(ind1, logical_not(ind2)))] = nan
        t_min[atleast_1d(logical_and(ind1, logical_not(ind2)))] = nan
    return vstack((t_min, t_max)), definiteRange
    
def defaultIntervalEngine(arg_lb_ub, fun, deriv, monotonity, convexity, \
                          criticalPoint = np.nan, criticalPointValue = np.nan):
    L, U, domain, definiteRange = arg_lb_ub.l, arg_lb_ub.u, arg_lb_ub.domain, arg_lb_ub.definiteRange
    R0 = arg_lb_ub.resolve()[0]
    assert R0.shape[0]==2, 'unimplemented yet'
    r_l, r_u = R0
    
    R2 = fun(R0)
    koeffs = (R2[1] - R2[0]) / (r_u - r_l)
    
    ind_eq = r_u == r_l
    
    Ld, Ud = L.d, U.d

    if monotonity == 1:
        new_l_resolved, new_u_resolved = R2
        U_dict, L_dict = Ud, Ld
        _argmin, _argmax = r_l, r_u
    elif monotonity == -1:
        new_u_resolved, new_l_resolved = R2
        U_dict, L_dict = Ld, Ud
        _argmin, _argmax = r_u, r_l
    else:
        ind = R2[1] > R2[0] 
        R2.sort(axis=0)
        new_l_resolved, new_u_resolved = R2
        
        _argmin = where(ind, r_l, r_u)
        _argmax = where(ind, r_u, r_l)
        if criticalPoint is not np.nan:
            ind_c = logical_and(r_l < criticalPoint, r_u > criticalPoint)
            if convexity == 1:
                new_l_resolved[ind_c] = criticalPointValue
                _argmin[ind_c] = criticalPoint
            elif convexity == -1:
                new_u_resolved[ind_c] = criticalPointValue
                _argmax[ind_c] = criticalPoint
        Keys = set().union(set(Ld.keys()), set(Ud.keys()))
        L_dict = dict((k, where(ind, Ld.get(k, 0), Ud.get(k, 0))) for k in Keys)
        U_dict = dict((k, where(ind, Ud.get(k, 0), Ld.get(k, 0))) for k in Keys)
            
    if convexity == -1:
        tmp2 = deriv(_argmax.view(multiarray)).view(ndarray).flatten()
        
        d_new = dict((v, tmp2 * val) for v, val in L_dict.items())
        U_new = surf(d_new, 0.0)
        U_new.c = new_u_resolved - U_new.maximum(domain)
        
        # for some simple cases
        if len(U_dict) >= 1:
            if any(ind_eq):
                koeffs[ind_eq] = tmp2[ind_eq]
            d_new = dict((v, koeffs * val) for v, val in U_dict.items())
            L_new = surf(d_new, 0.0)
            L_new.c = new_l_resolved -  L_new.minimum(domain)
        else:
            L_new = surf({}, new_l_resolved)                        
        R = boundsurf(L_new, U_new, definiteRange, domain)
    elif convexity == 1:
        tmp2 = deriv(_argmin.view(multiarray)).view(ndarray).flatten()
        d_new = dict((v, tmp2 * val) for v, val in L_dict.items())
        L_new = surf(d_new, 0.0)
        L_new.c = new_l_resolved - L_new.minimum(domain)
        
        # for some simple cases
        if len(U_dict) >= 1:
            if any(ind_eq):
                koeffs[ind_eq] = tmp2[ind_eq]
            d_new = dict((v, koeffs * val) for v, val in U_dict.items())
            U_new = surf(d_new, 0.0)
            U_new.c = new_u_resolved - U_new.maximum(domain)
        else:
            U_new = surf({}, new_u_resolved)
        R = boundsurf(L_new, U_new, definiteRange, domain)
    else:
        # linear oofuns with convexity = 0 calculate their intervals in other funcs
        raise FuncDesignerException('bug in FD kernel')
    return R, definiteRange
