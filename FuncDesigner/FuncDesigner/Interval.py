from numpy import ndarray, asscalar, isscalar, hstack, amax, amin, floor, ceil, pi, arange

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
    return [0.0] if arg_infinum < 0.0 < arg_supremum else []

#def IntegerCriticalPoints(arg_infinum, arg_supremum):
#    # TODO: check it for rounding errors
#    return arange(ceil(arg_infinum), ceil(1.0+arg_supremum), dtype=float).tolist()

def TrigonometryCriticalPoints(arg_infinum, arg_supremum):
    n1, n2 = int(floor(2 * arg_infinum / pi)), int(ceil(2 * arg_supremum / pi))
    # 6 instead of  5 for more safety, e.g. small numerical rounding effects
    return [i / 2.0 * pi for i in range(n1, amin((n1+6, n2))) if (arg_infinum < (i / 2.0) * pi <  arg_supremum)]
    
#def ufuncInterval(inp, domain, ufunc, criticalPointsFunc):
#    #inp = OOfun.input[0]
#    arg_infinum, arg_supremum = inp._interval(domain)
#    if not isscalar(arg_infinum) and arg_infinum.size > 1:
#        raise FuncDesignerException('not implemented for vectorized oovars yet')
#    tmp = ufunc(hstack([arg_infinum, arg_supremum] + ([] if criticalPointsFunc is False else criticalPointsFunc(arg_infinum, arg_supremum))))
#    return amin(tmp), amax(tmp)

#def ufuncInterval(OOfun, domain):
#    ufunc, criticalPointsFunc = OOfun.fun, OOfun.criticalPoints
#    inp = OOfun.input[0]
#    arg_infinum, arg_supremum = inp._interval(domain)
#    if not isscalar(arg_infinum) and arg_infinum.size > 1:
#        raise FuncDesignerException('not implemented for vectorized oovars yet')
#    tmp = ufunc(hstack([arg_infinum, arg_supremum] + ([] if criticalPointsFunc is None else criticalPointsFunc(arg_infinum, arg_supremum))))
#    return amin(tmp), amax(tmp)
