from numpy import ndarray, asscalar, isscalar, floor, pi, inf, nan, \
copy as Copy, logical_and, logical_or, where, asarray, any, all, atleast_1d, vstack, \
searchsorted, logical_not
import numpy as np
from boundsurf import boundsurf

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

def ZeroCriticalPointsInterval(inp, func):
    def interval(domain, dtype):
        is_abs = func == np.abs
        allowBoundSurf = is_abs
        lb_ub, definiteRange = inp._interval(domain, dtype, allowBoundSurf = allowBoundSurf)
        if is_abs and lb_ub.__class__ == boundsurf:
            return lb_ub.abs()
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

        return  vstack((t_min, t_max)), definiteRange
    return interval
#    if isscalar(arg_infinum):
#        return [0.0] if arg_infinum < 0.0 < arg_supremum else []
#    tmp = Copy(arg_infinum)
#    #tmp[where(logical_and(arg_infinum < 0.0, arg_supremum > 0.0))] = 0.0
#    tmp[atleast_1d(logical_and(arg_infinum < 0.0, arg_supremum > 0.0))] = 0.0
#    return [tmp]

def nonnegative_interval(inp, func, domain, dtype, F0, shift = 0.0):

    lb_ub, definiteRange = inp._interval(domain, dtype, allowBoundSurf = True)
    if type(lb_ub) == boundsurf:
        if 1 and func == np.sqrt:
            return lb_ub ** 0.5, definiteRange
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
        
        if type(lb2_ub2) == boundsurf:
            tmp2 = lb2_ub2.resolve()[0]
            t2_positive = all(tmp2 >= 0)
            t2_negative = all(tmp2 <= 0)
            if t2_positive or t2_negative:
                tmp1 = lb1_ub1.resolve()[0] if type(lb1_ub1) == boundsurf else lb1_ub1
                t1_positive = all(tmp1 >= 0)
                t1_negative = all(tmp1 <= 0)
                if (t1_positive or t1_negative) \
                and not any(logical_and(tmp1==0, np.isinf(tmp2)))\
                and not any(logical_and(tmp2==0, np.isinf(tmp1))):
                    # TODO: resolve that one that is better
                    #r = 0.99*lb1_ub1 * tmp2 + 0.01*lb2_ub2 * tmp1
                    #r = lb1_ub1 * tmp2 if nanmax(tmp2[1]-tmp2[0]) < nanmax(tmp1[1]-tmp1[0]) else lb2_ub2 * tmp1                    
                    # TODO: improve it
                    r = (lb1_ub1 if t1_positive else -lb1_ub1) * (lb2_ub2 if t2_positive else -lb2_ub2)
                    if t1_positive != t2_positive:
                        r = -r
            elif all(np.isfinite(tmp1)) and all(np.isfinite(tmp2)):
                r = 0.25 * ((lb1_ub1 + lb2_ub2) ** 2 - (lb1_ub1 - lb2_ub2) ** 2)
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

def pow_const_interval(self, other, domain, dtype):
    lb_ub, definiteRange = self._interval(domain, dtype, allowBoundSurf = True)
    if type(lb_ub) == boundsurf:
        lb_ub_resolved = lb_ub.resolve()[0]

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
    
