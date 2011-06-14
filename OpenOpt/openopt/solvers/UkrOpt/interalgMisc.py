from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take, argmax, argmin, abs, hstack, empty, insert, isfinite
from FuncDesigner import ooPoint

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    
def func10(y, e, vv):
    m, n = y.shape
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]

    r4 = 0.5 * (y + e)
    
    # TODO: remove the cycle
    #T1, T2 = tile(y, (2*n,1)), tile(e, (2*n,1))
    
    for i in range(n):
        t1, t2 = tile(y[:, i], 2*n), tile(e[:, i], 2*n)
        #t1, t2 = T1[:, i], T2[:, i]
        #T1[(n+i)*m:(n+i+1)*m, i] = T2[i*m:(i+1)*m, i] = r4[:, i]
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        LB[i], UB[i] = t1, t2

####        LB[i], UB[i] = T1[:, i], T2[:, i]

#    sh1, sh2, inds = [], [], []
#    for i in range(n):
#        sh1+= arange((n+i)*m, (n+i+1)*m).tolist()
#        inds +=  [i]*m
#        sh2 += arange(i*m, (i+1)*m).tolist()

#    sh1, sh2, inds = asdf(m, n)
#    asdf2(T1, T2, r4, sh1, sh2, inds)
    
    #domain = dict([(v, (T1[:, i], T2[:, i])) for i, v in enumerate(vv)])
    domain = dict([(v, (LB[i], UB[i])) for i, v in enumerate(vv)])
    
    domain = ooPoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    return domain

def func8(domain, func, dataType):
    TMP = func.interval(domain, dataType)
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType)

def getr4Values(domain, func, C, contol, dataType):
    #TODO: remove 0.5*(val[0]+val[1]) from cycle
    cs = dict([(key, 0.5*(val[0]+val[1])) for key, val in domain.items()])
    cs = ooPoint(cs, skipArrayCast = True)
    cs.isMultiPoint = True
    
    # TODO: improve it
    m = domain.values()[0][0].size
    
    r15 = empty(m, bool)
    r15.fill(True)
    for f, r16, r17 in C:
        c = f(cs)
        ind = logical_and(c + contol > r16, c - contol < r17)
        r15 = logical_and(r15, ind)
    if not all(r15):
        F = empty(m, dataType)
        F.fill(nan)
        cs = dict([(key, 0.5*(val[0][r15]+val[1][r15])) for key, val in domain.items()])
        cs = ooPoint(cs, skipArrayCast = True)
        cs.isMultiPoint = True
        F[r15] = func(cs)
        r = atleast_1d(F)
    else:
        r = atleast_1d(func(cs))
    return r

def r2(r3, domain, dataType):
    r23 = nanargmin(r3)
    if isnan(r23):
        r23 = 0
    
    # TODO: check it , maybe it can be improved
    #bestCenter = cs[r23]
    r7 = array([0.5*(val[0][r23]+val[1][r23]) for val in domain.values()], dtype=dataType)
    r8 = atleast_1d(r3)[r23] if not isnan(r23) else inf
    return r7, r8
    
def func7(y, e, o, a, FV, _s):
    r10 = logical_and(all(isnan(o), 1), all(isnan(a), 1))
    if any(r10):
        j = where(logical_not(r10))[0]
        lj = j.size
        y = take(y, j, axis=0, out=y[:lj])
        e = take(e, j, axis=0, out=e[:lj])
        o = take(o, j, axis=0, out=o[:lj])
        a = take(a, j, axis=0, out=a[:lj])
        FV = take(FV, j, axis=0, out=FV[:lj])
        _s = _s[j]
    return y, e, o, a, FV, _s

def func9(an, fo, g):
    maxo = [node.key for node in an]
    ind = searchsorted(maxo, fo, side='right')
    if ind == len(maxo):
        return an, g
    else:
        g = nanmin((g, nanmin(atleast_1d(maxo[ind]))))
        return an[:ind], g


def func5(an, nn, g):
    m = len(an)
    if m > nn:
        maxo = [node.key for node in an]
        g = nanmin((maxo[nn], g))
        an = an[:nn]
    return an, g

def func4(y, e, o, a, fo):
    cs = 0.5*(y + e)
    n = y.shape[1]
    o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
    ind = logical_or(o_modL > fo, isnan(o_modL)) # TODO: assert isnan(o_modL) is same to isnan(a_modL)
    if any(ind):
        y[ind] = cs[ind]
    ind = logical_or(o_modU > fo, isnan(o_modU))# TODO: assert isnan(o_modU) is same to isnan(a_modU)
    if any(ind):
        e[ind] = cs[ind]
    return y, e

def func3(an, maxActiveNodes):
    m = len(an)
    if m > maxActiveNodes:
        an1, _in = an[:maxActiveNodes], an[maxActiveNodes:]
    else:
        an1, _in = an, array([], object)
    return an1, _in

def func1(y, e, o, a, varTols, _s_prev, r9 = None):
    m, n = y.shape
    w = arange(m)
    #a = a.copy() # to remain it unchanged in higher stack level funcs
    #a[e-y<varTols] = nan
    
    #1
    _s = func13(o, a)
    
    #2
    #_s = nanmin(a, 1)
    
    #3
    #_s = nanmax(a, 1)
    
    #1
    t = nanargmin(a, 1) % n

    #t = nanargmin(o, 1) % n
    
    #2
#    U1, U2 = a[:, :n].copy(), a[:, n:] 
##    #TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
#    U1 = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
#    t = nanargmin(U1, 1)

    #3
#    L1, L2, U1, U2 = o[:, :n], o[:, n:], a[:, :n], a[:, n:] 
#    U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
#    L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
#    t = nanargmin(U-L, 1)

    
#    tmp = a[(arange(m), r10)].reshape(-1, 1).copy()
#    a[:, r10] = nan
    
    #a[(arange(m), r10)] = nan
    #ind = logical_or(any(a == tmp, 1), all(isinf(a), 1))
    # TODO: improve it, handle the number as solver parameter
    
    #ind = logical_or(all(isinf(a), 1), _s * (1.0 + max((1e-15, 2.0 ** (-n)))) >= _s_prev)
    
    #1
    #ind = _s * (1.0 + max((1e-15, 2 ** (-n)))) >= _s_prev
    
    #2
    d = nanmax([a[w, t] - o[w, t], 
                a[w, n+t] - o[w, n+t]], 0)

    ind = d * (1.0 + max((1e-15, 2 ** (-n)))) >= _s_prev
    
    #print _s_prev, '\n', _s
#    print nanmax(_s_prev), nanmax(_s)
#    print '\n'
#    print '2:', nanmax(_s / _s_prev), nanmax(_s), nanmax(_s_prev)
#    print '3:', nanmin(nanmax(U1-L1, 1))
    
    if r9 is not None:
        ind = logical_or(ind, r9)

    if any(ind):
        bs = e[ind] - y[ind]
        t[ind] = nanargmax(bs, 1) # ordinary numpy.argmax can be used as well
    #a[:, r10] = tmp
        
    return t, _s
    
def func13(o, a): 
    m, n = o.shape
    n /= 2
    case = 2
    if case == 1:
        U1, U2 = a[:, :n].copy(), a[:, n:] 
        #TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
        U1 = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
        return nanmin(U1, 1)
    elif case == 2:
        L1, L2, U1, U2 = o[:, :n], o[:, n:], a[:, :n], a[:, n:] 
        U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
        L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
        return nanmax(U-L, 1)

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


def func12(an, maxActiveNodes, maxSolutions, solutions, r6, varTols, fo):
    if len(an) == 0:
        return array([]), array([]), array([]), array([])
    _in = an
    if r6.size != 0:
        r11, r12 = r6 - varTols, r6 + varTols
    y, e, S = [], [], []
    N = 0

    while True:
        an1Candidates, _in = func3(_in, maxActiveNodes)

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
            ind = logical_and(all(yc >= r11[i], 1), all(ec <= r12[i], 1))
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
        if len(_in) == 0 or N >= maxActiveNodes: 
            y, e, _s = vstack(y), vstack(e), hstack(S)
            break
    return y, e, _in, _s

def r13(y, e, r3, fTol, varTols, solutions, r6):
    n = r3.shape[1] / 2
    r18, r19 = r3[:, :n], r3[:, n:]
    r5_L, r5_U =  where(r18 < fTol), where(r19 < fTol)
    r4 = 0.5 * (y + e)
    r20 = 0.5 * (e - y)
    r5 = []
    # L
    r21 = r4[r5_L[0]].copy()
    for I in range(len(r5_L[0])):#TODO: rework it
        i, j = r5_L[0][I], r5_L[1][I]
        tmp = r4[i].copy()
        tmp[j] -= 0.5*r20[i, j]
        r5.append(tmp)
    # U
    r22 = r4[r5_U[0]].copy()
    for I in range(len(r5_U[0])):#TODO: rework it
        i, j = r5_U[0][I], r5_U[1][I]
        tmp = r4[i].copy()
        tmp[j] += 0.5*r20[i, j]
        r5.append(tmp)
    
    for c in r5:
        if len(solutions) == 0 or not any(all(abs(c - r6) < varTols, 1)): 
            solutions.append(c)
            r6 = asarray(solutions)
            
    return solutions, r6

def r14(p, y, e, vv, asdf1, C, CBKPMV, itn, g, nNodes,  \
                 frc, fTol, maxSolutions, varTols, solutions, r6, _in, dataType, isSNLE, \
                 maxNodes, _s, xRecord):
    ip = func10(y, e, vv)
        
    o, a = func8(ip, asdf1, dataType)
    if p.debug and any(a + 1e-15 < o):  
        p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
    if p.debug and any(logical_xor(isnan(o), isnan(a))):
        p.err('bug in FuncDesigner intervals engine')
    
    r3 = getr4Values(ip, asdf1, C, p.contol, dataType) 

    xk, Min = r2(r3, ip, dataType)
    
    if CBKPMV > Min:
        CBKPMV = Min
        xRecord = xk# TODO: is copy required?
    if frc > Min:
        frc = Min
    
    fo = 0.0 if isSNLE else min((frc, CBKPMV - (fTol if maxSolutions == 1 else 0.0))) 
    #print itn,  y.shape[0], fo
    m, n = e.shape
    o, a, r3 = o.reshape(2*n, m).T, a.reshape(2*n, m).T, r3.reshape(2*n, m).T
    if itn == 0: 
        _s = atleast_1d(nanmax(a))
    y, e, o, a, r3, _s = func7(y, e, o, a, r3, _s)

    nodes = func11(y, e, o, a, _s, r3) if maxSolutions != 1 else func11(y, e, o, a, _s)
    nodes.sort(key = lambda obj: obj.key)

    if len(_in) == 0:
        an = nodes
    else:
        arr1 = [node.key for node in _in]
        arr2 = [node.key for node in nodes]
        r10 = searchsorted(arr1, arr2)
        an = insert(_in, r10, nodes)
    
    if maxSolutions != 1:
        solutions, r6 = r13(y, e, r3, fTol, varTols, solutions, r6)
        
        p._nObtainedSolutions = len(solutions)
        if p._nObtainedSolutions > maxSolutions:
            solutions = solutions[:maxSolutions]
            p.istop = 0
            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
            return an, g, fo, None, solutions, r6, xRecord, frc, CBKPMV
                
    
    p.iterfcn(xk)
    if p.istop != 0: 
        return an, g, fo, None, solutions, r6, xRecord, frc, CBKPMV
    if isSNLE and maxSolutions == 1 and Min <= fTol:
        # TODO: rework it for nonlinear systems with non-bound constraints
        p.istop, p.msg = 1000, 'required solution has been obtained'
        return an, g, fo, None, solutions, r6, xRecord, frc, CBKPMV
    
    
    an, g = func9(an, fo, g)

    nn = 1 if asdf1.isUncycled and all(isfinite(a)) and all(isfinite(o)) and p._isOnlyBoxBounded else maxNodes
    
    an, g = func5(an, nn, g)
    nNodes.append(len(an))
    return an, g, fo, _s, solutions, r6, xRecord, frc, CBKPMV


def func11(y, e, o, a, _s, r3 = None): 
    m, n = y.shape
    o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
    Tmp = nanmax(where(o_modU<o_modL, o_modU, o_modL), 1)
    if r3 is None:
        return [si(Tmp[i], y[i], e[i], o[i], a[i], _s[i]) for i in range(m)]
    else:
        r18, r19 = r3[:, :n], r3[:, n:]
        return [si(Tmp[i], y[i], e[i], o[i], a[i], _s[i], r18[i], r19[i]) for i in range(m)]

class si:
    fields = ['key', 'y', 'e', 'o', 'a', '_s','r18', 'r19']
    def __init__(self, *args):
        for i in range(len(args)):
            setattr(self, self.fields[i], args[i])
    
#    __lt__ = lambda self, other: self.key < other.key
#    __le__ = lambda self, other: self.key <= other.key
#    __gt__ = lambda self, other: self.key > other.key
#    __ge__ = lambda self, other: self.key >= other.key
#    __eq__ = lambda self, other: self.key == other.key
