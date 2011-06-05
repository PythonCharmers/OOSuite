from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take, argmax, argmin, abs, hstack
from FuncDesigner import ooPoint

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    
def func10(y, e, vv):
    m, n = y.shape
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]

    Centers = 0.5 * (y + e)
    
    # TODO: remove the cycle
    #T1, T2 = tile(y, (2*n,1)), tile(e, (2*n,1))
    
    for i in range(n):
        t1, t2 = tile(y[:, i], 2*n), tile(e[:, i], 2*n)
        #t1, t2 = T1[:, i], T2[:, i]
        #T1[(n+i)*m:(n+i+1)*m, i] = T2[i*m:(i+1)*m, i] = Centers[:, i]
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = Centers[:, i]
        LB[i], UB[i] = t1, t2

####        LB[i], UB[i] = T1[:, i], T2[:, i]

#    sh1, sh2, inds = [], [], []
#    for i in range(n):
#        sh1+= arange((n+i)*m, (n+i+1)*m).tolist()
#        inds +=  [i]*m
#        sh2 += arange(i*m, (i+1)*m).tolist()

#    sh1, sh2, inds = asdf(m, n)
#    asdf2(T1, T2, Centers, sh1, sh2, inds)
    
    #domain = dict([(v, (T1[:, i], T2[:, i])) for i, v in enumerate(vv)])
    domain = dict([(v, (LB[i], UB[i])) for i, v in enumerate(vv)])
    
    domain = ooPoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    return domain

def func8(domain, func, dataType):
    TMP = func.interval(domain, dataType)
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType)

def getCentersValues(domain, func, dataType):
    #TODO: remove 0.5*(val[0]+val[1]) from cycle
    cs = dict([(key, 0.5*(val[0]+val[1])) for key, val in domain.items()])
    cs = ooPoint(cs, skipArrayCast = True)
    cs.isMultiPoint = True
    return atleast_1d(func(cs))

def getBestCenterAndObjective(FuncVals, domain, dataType):
    bestCenterInd = nanargmin(FuncVals)
    if isnan(bestCenterInd):
        bestCenterInd = 0
        bestCenterObjective = inf
    
    # TODO: check it , maybe it can be improved
    #bestCenter = cs[bestCenterInd]
    bestCenterCoords = array([0.5*(val[0][bestCenterInd]+val[1][bestCenterInd]) for val in domain.values()], dtype=dataType)
    bestCenterObjective = atleast_1d(FuncVals)[bestCenterInd]
    return bestCenterCoords, bestCenterObjective
    
def func7(y, e, o, a, FV, _s):
    IND = logical_and(all(isnan(o), 1), all(isnan(a), 1))
    if any(IND):
        j = where(logical_not(IND))[0]
        lj = j.size
        y = take(y, j, axis=0, out=y[:lj])
        e = take(e, j, axis=0, out=e[:lj])
        o = take(o, j, axis=0, out=o[:lj])
        a = take(a, j, axis=0, out=a[:lj])
        FV = take(FV, j, axis=0, out=FV[:lj])
        _s = _s[j]
    return y, e, o, a, FV, _s

def func9(an, fo, g):
    ar = [node.key for node in an]
    ind = searchsorted(ar, fo, side='right')
    if ind == len(ar):
        return an, g
    else:
        g = nanmin((g, nanmin(atleast_1d(ar[ind]))))
        return an[:ind], g


def func5(an, nn, g):
    m = len(an)
    if m > nn:
        ar = [node.key for node in an]
        g = nanmin((ar[nn], g))
        an = an[:nn]
    return an, g

def func4(y, e, o, a, fo):
    cs = 0.5*(y + e)
    n = y.shape[1]
    s, q = o[:, 0:n], o[:, n:2*n]
    ind = logical_or(s > fo, isnan(s)) # TODO: assert isnan(s) is same to isnan(a_modL)
    if any(ind):
        y[ind] = cs[ind]
    ind = logical_or(q > fo, isnan(q))# TODO: assert isnan(q) is same to isnan(a_modU)
    if any(ind):
        e[ind] = cs[ind]
    return y, e

def func3(an, mn):
    m = len(an)
    if m > mn:
        an1, _in = an[:mn], an[mn:]
    else:
        an1, _in = an, []
    return an1, _in

def func1(y, e, o, a, varTols, _s_prev, CoordDivisionInd = None):
    m, n = y.shape
    #a = a.copy() # to remain it unchanged in higher stack level funcs
    #a[e-y<varTols] = nan
    _s = func13(o, a)
    
    IND = nanargmin(a, 1)

#    IND = nanargmin(o-a, 1)
#    _s = (o-a)[(arange(m), IND)]
    
    #print '-'*10
    
    #assert all(_s_prev == _s) or all(_s_prev + 1e-10*abs(_s) >= _s  - 1e-15)
    
#    print _s_prev
#    print _s
    #print '='*10
    
    t = IND % n
    
#    tmp = a[(arange(m), IND)].reshape(-1, 1).copy()
#    a[:, IND] = nan
    
    #a[(arange(m), IND)] = nan
    #ind = logical_or(any(a == tmp, 1), all(isinf(a), 1))
    # TODO: improve it, handle the number as solver parameter
    
    ind = logical_or(all(isinf(a), 1), _s_prev  <= _s+  _s_prev / 2.0 ** n)
    
    if CoordDivisionInd is not None:
        ind = logical_or(ind, CoordDivisionInd)
    
    if any(ind):
        bs = e[ind] - y[ind]
        t[ind] = nanargmax(bs, 1) # ordinary numpy.argmax can be used as well
    #a[:, IND] = tmp
        
    return t, _s
    
def func13(o, a): 
    m, n = o.shape
    n /= 2
    case = 2
    if case == 1:
        U1, U2 = a[:, :n].copy(), a[:, n:] 
        ind = U2>U1#TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
        U1[ind] = U2[ind]
        return nanmin(U1, 1)
    elif case == 2:
        L1, L2, U1, U2 = o[:, :n], o[:, n:], a[:, :n], a[:, n:] 
        U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
        L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
        return nanmin(U-L, 1)

def func2(y, e, t):
    new_y, en = y.copy(), e.copy()
    m = y.shape[0]
    w = arange(m)
    th = 0.5 * (new_y[w, t] + en[w, t])
    new_y[w, t] = th
    en[w, t] = th
    
    new_y = vstack((y, new_y))
    en = vstack((en, e))
    
    return new_y, en


def func12(an, mn, maxSolutions, solutions, SolutionCoords, varTols, fo, _s_prev):
    if len(an) == 0:
        return [], [], [], []
    _in = an
    if SolutionCoords is not None:
        solutionCoordsL, solutionCoordsU = SolutionCoords - varTols, SolutionCoords + varTols
    y, e, S = [], [], []
    N = 0

    while True:
        an1Candidates, _in = func3(_in, mn)

        yc, ec, oc, ac, SIc = asarray([t.y for t in an1Candidates]), \
        asarray([t.e for t in an1Candidates]), \
        asarray([t.o for t in an1Candidates]), \
        asarray([t.a for t in an1Candidates]), \
        asarray([t._s for t in an1Candidates])
        
        yc, ec = func4(yc, ec, oc, ac, fo)
        t, _s = func1(yc, ec, oc, ac, varTols, SIc)
        yc, ec = func2(yc, ec, t)
        _s = tile(_s, 2)
        
        if maxSolutions == 1 or len(solutions) == 0: 
            y, e = yc, ec
            break
        
        for i in range(len(solutions)):
            ind = logical_and(all(yc >= solutionCoordsL[i], 1), all(ec <= solutionCoordsU[i], 1))
            if any(ind):
                j = where(logical_not(ind))[0]
                lj = j.size
                yc = take(yc, j, axis=0, out=yc[:lj])
                ec = take(ec, j, axis=0, out=ec[:lj])
                _s = _s[j]
        y.append(yc)
        e.append(ec)
        S.append(_s)
        N += yc.shape[0]
        if len(_in) == 0 or N >= mn: 
            y, e, _s = vstack(y), vstack(e), hstack(S)
            break
    return y, e, _in, _s

def func11(y, e, o, a, _s, FCl = None, FCu = None): 
    m, n = y.shape
    s, q = o[:, 0:n], o[:, n:2*n]
    Tmp = nanmax(where(q<s, q, s), 1)
    if FCl is None:
        return [si(Tmp[i], y[i], e[i], o[i], a[i], _s[i]) for i in range(m)]
    else:
        return [si(Tmp[i], y[i], e[i], o[i], a[i], _s[i], FCl[i], FCu[i]) for i in range(m)]

class si:
    fields = ['key', 'y', 'e', 'o', 'a', '_s','FCl', 'FCu']
    def __init__(self, *args):
        for i in range(len(args)):
            setattr(self, self.fields[i], args[i])
    
#    __lt__ = lambda self, other: self.key < other.key
#    __le__ = lambda self, other: self.key <= other.key
#    __gt__ = lambda self, other: self.key > other.key
#    __ge__ = lambda self, other: self.key >= other.key
#    __eq__ = lambda self, other: self.key == other.key
