from numpy import ndarray, asscalar, isscalar, hstack, amax, amin, floor, ceil, pi, \
arange, copy as Copy, logical_and, where, asarray, any, atleast_1d, empty_like, nan, logical_not

class Interval:
    def __init__(self, l, u):
        if isinstance(l, ndarray) and l.size == 1: l = asscalar(l)
        if isinstance(u, ndarray) and u.size == 1: u = asscalar(u)
        self.lb, self.ub = l, u
    def __str__(self):
        return 'FuncDesigner interval with lower bound %s and upper bound %s' % (self.lb, self.ub)
    def __repr__(self):
        return str(self)


def ZeroCriticalPoints(arg_infinum, arg_supremum):
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
def TrigonometryCriticalPoints(arg_infinum, arg_supremum):
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
        tmp[atleast_1d(ind)] = (arrN[ind]+i)*pi/2
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

def nonnegative_interval(inp, func, domain, dtype):
    lb, ub = inp._interval(domain, dtype)
    ind = lb < 0.0
    if any(ind):
        t_min, t_max = atleast_1d(empty_like(lb)), atleast_1d(empty_like(ub))
        t_min.fill(nan)
        t_max.fill(nan)
        ind2 = ub >= 0
        t_min[atleast_1d(logical_not(ind))] = func(lb)
        t_min[atleast_1d(logical_and(ind2, ind))] = 0.0
        t_max[atleast_1d(ind2)] = func(ub[ind2])
        t_min, t_max = func(lb), func(ub)
        t_min[atleast_1d(logical_and(lb < 0, ub > 0))] = 0.0
    else:
        t_min, t_max = func(lb), func(ub)
    return t_min, t_max
