from numpy import ndarray, asscalar, isscalar, floor, pi, \
copy as Copy, logical_and, where, asarray, any, atleast_1d, vstack, \
searchsorted
import numpy as np

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

#def TrigonometryCriticalPoints2(arg_infinum, arg_supremum):
#    n1, n2 = int(floor(2 * arg_infinum / pi)), int(ceil(2 * arg_supremum / pi))
#    # 6 instead of  5 for more safety, e.g. small numerical rounding effects
#    return [i / 2.0 * pi for i in range(n1, amin((n1+6, n2))) if (arg_infinum < (i / 2.0) * pi <  arg_supremum)]

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
        lb_ub, definiteRange = inp._interval(domain, dtype)
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
    lb_ub, definiteRange = inp._interval(domain, dtype)
    lb, ub = lb_ub[0], lb_ub[1]
    
    t_min_max = func(lb_ub)
    th = shift # 0.0 + shift = shift
    ind = lb < th
    if any(ind):
        t_min_max[0][atleast_1d(logical_and(lb < th, ub >= th))] = F0
        if definiteRange is not False:
            if definiteRange is True:
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
