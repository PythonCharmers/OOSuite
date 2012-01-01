from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take, argmax, argmin, abs, hstack, empty, insert, isfinite, append, atleast_2d, \
prod, sqrt, int32, int64, log2, log
from FuncDesigner import oopoint
from interalgT import *

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax

def func82(y, e, vv, f, dataType):
    domain = dict([(v, (y[:, i], e[:, i])) for i, v in enumerate(vv)])
    r, r0 = f.iqg(domain, dataType)
    dep = f._getDep() # TODO: Rework it for fixed vars
    o_l, o_u, a_l, a_u = [], [], [], []
    definiteRange = True
    for v in vv:
        # !!!! TODO: rework and optimize it
        if v in dep:
            o_l.append(r[v][0].lb)
            o_u.append(r[v][1].lb)
            a_l.append(r[v][0].ub)
            a_u.append(r[v][1].ub)
            definiteRange = logical_and(definiteRange, r[v][0].definiteRange)
            definiteRange = logical_and(definiteRange, r[v][1].definiteRange)
        else:
            o_l.append(r0.lb)
            o_u.append(r0.lb)
            a_l.append(r0.ub)
            a_u.append(r0.ub)
            definiteRange = logical_and(definiteRange, r0.definiteRange)
        o, a = hstack(o_l+o_u), hstack(a_l+a_u)    
    return o, a, definiteRange

def func10(y, e, vv):
    m, n = y.shape
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]

    r4 = (y + e) / 2
    
    # TODO: remove the cycle
    #T1, T2 = tile(y, (2*n,1)), tile(e, (2*n,1))
    
    for i in range(n):
        t1, t2 = tile(y[:, i], 2*n), tile(e[:, i], 2*n)
        #t1, t2 = T1[:, i], T2[:, i]
        #T1[(n+i)*m:(n+i+1)*m, i] = T2[i*m:(i+1)*m, i] = r4[:, i]
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        
        if vv[i].domain is bool:
            indINQ = y[:, i] != e[:, i]
            tmp = t1[(n+i)*m:(n+i+1)*m]
            tmp[indINQ] = 1
            tmp = t2[i*m:(i+1)*m]
            tmp[indINQ] = 0
            
#        if vv[i].domain is bool:
#            t1[(n+i)*m:(n+i+1)*m] = 1
#            t2[i*m:(i+1)*m] = 0
#        else:
#            t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        
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
    
    domain = oopoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    return domain

def func8(domain, func, dataType):
    TMP = func.interval(domain, dataType)
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType), TMP.definiteRange

def getr4Values(vv, y, e, tnlh, func, C, contol, dataType, p):
    n = y.shape[1]
    # TODO: rework it wrt nlh
    #cs = dict([(key, asarray((val[0]+val[1])/2, dataType)) for key, val in domain.items()])
    if tnlh is None:
        cs = dict([(oovar, asarray((y[:, i]+e[:, i])/2, dataType)) for i, oovar in enumerate(vv)])
    else:
        tnlh = tnlh.copy()
        tnlh[atleast_1d(tnlh<1e-300)] = 1e-300 # to prevent division by zero
        tnlh[atleast_1d(isnan(tnlh))] = inf #- check it!
        tnlh_l_inv, tnlh_u_inv = 1.0 / tnlh[:, :n], 1.0 / tnlh[:, n:]
        wr4 = (y * tnlh_l_inv + e * tnlh_u_inv) / (tnlh_l_inv + tnlh_u_inv)
        ind = tnlh_l_inv == tnlh_u_inv # especially important for tnlh_l_inv == tnlh_u_inv = 0
        wr4[ind] = (y[ind] + e[ind]) / 2
#        wr4 = (y + e) / 2
        cs = dict([(oovar, asarray(wr4[:, i], dataType)) for i, oovar in enumerate(vv)])
        
        #OLD
#        cs = dict([(oovar, asarray((y[:, i]+e[:, i])/2, dataType)) for i, oovar in enumerate(vv)])
#        wr4 = (y + e) / 2

        
    cs = oopoint(cs, skipArrayCast = True)
    cs.isMultiPoint = True
    
    # TODO: improve it
    #V = domain.values()
    #m = V[0][0].size if type(V) == list else next(iter(V))[0].size
    m = y.shape[0]
    if len(C) != 0:
        r15 = empty(m, bool)
        r15.fill(True)
        for f, r16, r17 in C:
            c = f(cs)
            ind = logical_and(c  >= r16, c <= r17) # here r16 and r17 are already shifted by required tolerance
            r15 = logical_and(r15, ind)
    else:
        r15 = True
    if not any(r15):
        F = empty(m, dataType)
        F.fill(2**31-2 if dataType in (int32, int64, int) else nan) 
    elif all(r15):
        F = func(cs)
    else:
        #cs = dict([(oovar, (y[r15, i] + e[r15, i])/2) for i, oovar in enumerate(vv)])
        #cs = ooPoint(cs, skipArrayCast = True)
        cs.isMultiPoint = True
        tmp = func(cs)
        F = empty(m, dataType)
        #F.fill(nanmax(tmp)+1) 
        F.fill(2**31-15 if dataType in (int32, int64, int) else nan)
        F[r15] = tmp[r15]
    return atleast_1d(F), ((y+e) / 2 if tnlh is None else wr4)

def r2(PointVals, PointCoords, dataType):
    r23 = nanargmin(PointVals)
    if isnan(r23):
        r23 = 0
    # TODO: check it , maybe it can be improved
    #bestCenter = cs[r23]
    #r7 = array([(val[0][r23]+val[1][r23]) / 2 for val in domain.values()], dtype=dataType)
    #r8 = atleast_1d(r3)[r23] if not isnan(r23) else inf
    r7 = array(PointCoords[r23], dtype=dataType)
    r8 = atleast_1d(PointVals)[r23] 
    return r7, r8
    
def func3(an, maxActiveNodes):
    m = len(an)
    if m > maxActiveNodes:
        an1, _in = an[:maxActiveNodes], an[maxActiveNodes:]
    else:
        an1, _in = an, array([], object)
    return an1, _in

def func1(tnlhf, tnlhf_curr, y, e, o, a, _s_prev, p, Case, r9 = None):
    m, n = y.shape
    w = arange(m)

    if Case != 'IP':
        if p.solver.dataHandling == 'sorted':
            _s = func13(o, a, Case)
            t = nanargmin(a, 1) % n
            d = nanmax([a[w, t] - o[w, t], 
                    a[w, n+t] - o[w, n+t]], 0)
            
            ## !!!! Don't replace it by (_s_prev /d- 1) to omit rounding errors ###
            #ind = 2**(-n) >= (_s_prev - d)/asarray(d, 'float64')
            
            #NEW
            ind = d  >=  _s_prev / 2 ** (1.0e-12/n)
            #ind.fill(False)
            ###################################################
        elif p.solver.dataHandling == 'raw':
            #tnlhf = tnlhf_curr 
            tnlh_1, tnlh_2 = tnlhf[:, 0:n], tnlhf[:, n:]
            PointCoordsTNHLF_max =  where(logical_or(tnlh_1 < tnlh_2, isnan(tnlh_1)), tnlh_2, tnlh_1)
            TNHLF_min =  where(logical_or(tnlh_1 > tnlh_2, isnan(tnlh_1)), tnlh_2, tnlh_1)
            _s = nanmin(TNHLF_min, 1)
            
            tnlh_curr_1, tnlh_curr_2 = tnlhf_curr[:, 0:n], tnlhf_curr[:, n:]
            TNHL_curr_min =  where(logical_or(tnlh_curr_1 < tnlh_curr_2, isnan(tnlh_curr_2)), tnlh_curr_1, tnlh_curr_2)
            #1
            t = nanargmin(TNHL_curr_min, 1)
            
            t = nanargmin(TNHLF_min, 1)
            #2
            #t = nanargmin(tnlhf, 1) % n
            #3
            #t = nanargmin(a, 1) % n
            
            d = nanmin(vstack(([tnlhf[w, t], tnlhf[w, n+t]])), 0)
            _s = d

            #OLD
            #!#!#!#! Don't replace it by _s_prev - d <= ... to omit inf-inf = nan !#!#!#
            #ind = _s_prev  <= d + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            #ind = _s_prev - d <= ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            
            #NEW
            ind = _s_prev  <= d + 1.0/n#(n*(1+p.nc+p.nh+p.nb+p.nbeq)) 
            
            #print _s_prev - d
            ###################################################
            #d = ((tnlh[w, t]* tnlh[w, n+t])**0.5)
        else:
            assert 0
    else: # IP
        tmp = a[:, 0:n]-o[:, 0:n]+a[:, n:]-o[:, n:]
        #_s = nanmax(tmp, 1)
        _s = 0.5*nanmin(tmp, 1)
        t = nanargmin(tmp,1)
        #d = tmp[w, t]
        ind = 2**(-n) >= (_s_prev - _s)/asarray(_s, 'float64')
        #ind = _s  >= 2 ** (1.0/n) * _s_prev
        
#        if p.debug: 
#            assert all(_s_prev >= _s)
        #print len(where(ind)[0]), len(d)
        #ind = d  >= 2 ** (1.0/n) * _s_prev
    
    #ind = d * (1.0 + max((1e-15, 2 ** (-n)))) >= _s_prev
    
    if r9 is not None:
        ind = logical_or(ind, r9)
    
    if any(ind):
        r10 = where((_s_prev -d)*n <=  1.0)
#        print _s_prev
#        print ((_s_prev -d)*n)[r10]
#        print('ind length: %d' % len(where(ind)[0]))
        bs = e[ind] - y[ind]
        t[ind] = nanargmax(bs, 1) # ordinary numpy.argmax can be used as well
        
    return t, _s
    
def func13(o, a, case = 2): 
    m, n = o.shape
    n /= 2
    if case == 1:
        U1, U2 = a[:, :n].copy(), a[:, n:] 
        #TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
        U1 = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
        return nanmin(U1, 1)
        
    L1, L2, U1, U2 = o[:, :n], o[:, n:], a[:, :n], a[:, n:] 
    if case == 2:
        U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
        L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
        return nanmax(U-L, 1)
#    elif case == 'IP': # IP
#        return nanmax(U1-L1+U2-L2, 1)
    else: 
        raise('bug in interalg kernel')

def func2(y, e, t, vv):
    new_y, new_e = y.copy(), e.copy()
    m, n = y.shape
    w = arange(m)
    
    # TODO: omit or imporove it for all-float problems    
    th = (new_y[w, t] + new_e[w, t]) / 2
    BoolVars = [v.domain is bool for v in vv]
    if any(BoolVars):
        indBool = where(BoolVars)[0]
        if len(indBool) != n:
            new_y[w, t] = th
            new_e[w, t] = th
            new_y[indBool, t] = 1
            new_e[indBool, t] = 0
        else:
            new_y[w, t] = 1
            new_e[w, t] = 0
    else:
        new_y[w, t] = th
        new_e[w, t] = th
    
    new_y = vstack((y, new_y))
    new_e = vstack((new_e, e))
    
    return new_y, new_e


def func12(an, maxActiveNodes, p, solutions, r6, vv, varTols, fo, Case):
    if len(an) == 0:
        return array([]), array([]), array([]), array([])
    _in = an
    if r6.size != 0:
        r11, r12 = r6 - varTols, r6 + varTols
    y, e, S = [], [], []
    N = 0
    maxSolutions = p.maxSolutions
    
    while True:
        an1Candidates, _in = func3(_in, maxActiveNodes)

        yc, ec, oc, ac, SIc = asarray([t.y for t in an1Candidates]), \
        asarray([t.e for t in an1Candidates]), \
        asarray([t.o for t in an1Candidates]), \
        asarray([t.a for t in an1Candidates]), \
        asarray([t._s for t in an1Candidates])
        
        
        
        tnlhf = asarray([t.tnlhf for t in an1Candidates]) if p.solver.dataHandling == 'raw' else None
        tnlhf_curr = asarray([t.tnlh_curr for t in an1Candidates]) if p.solver.dataHandling == 'raw' else None
        
        if p.probType != 'IP': 
            nlhc = asarray([t.nlhc for t in an1Candidates])
            yc, ec = func4(yc, ec, oc, ac, nlhc, fo)
        t, _s = func1(tnlhf, tnlhf_curr, yc, ec, oc, ac, SIc, p, Case)
        
        # OLD
        #_s = tile(_s, 2)
        
        # NEW
        if p.probType != 'IP': 
            _s = tile(_s, 2)
        else:
            m, n = yc.shape
            #w = arange(m)
            #t = nanargmin(ac[:, :n]-oc[:, :n] + ac[:, n:] - oc[:, n:], 1) 
            #tmp = nanmax([ac[:, :n] - oc[:, :n], 
            #        ac[:, n:] - oc[:, n:]], 0)
#            tmp = ac[:, :n] - oc[:, :n] + ac[:, n:] - oc[:, n:]
#            tmp = nanmin(tmp, 1)
##            tmp = ac[w, t] - oc[w, t] + \
##                    ac[w, n+t] - oc[w, n+t]
#            _s = tile(2*tmp, 2)
            oc_modL, oc_modU = oc[:, :n], oc[:, n:]
            ac_modL, ac_modU = ac[:, :n], ac[:, n:]
            # TODO: handle nans
            mino = where(oc_modL < oc_modU, oc_modL, oc_modU)
            maxa = where(ac_modL < ac_modU, ac_modU, ac_modL)
            _s = nanmin(maxa - mino, 1)
            _s = tile(_s, 2)

        yc, ec = func2(yc, ec, t, vv)

        if maxSolutions == 1 or len(solutions) == 0: 
            y, e = yc, ec
            break
        
        # TODO: change cycle variable if len(solutions) >> maxActiveNodes
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

Fields = ['key', 'y', 'e', 'nlhf','nlhc', 'o', 'a', '_s']
#FuncValFields = ['key', 'y', 'e', 'nlhf','nlhc', 'o', 'a', '_s','r18', 'r19']
IP_fields = ['key', 'y', 'e', 'o', 'a', '_s','F', 'volume', 'volumeResidual']

def func11(y, e, nlhc, o, a, _s, p): 
    m, n = y.shape
    if p.probType == "IP":
        w = arange(m)
        # TODO: omit recalculation from func1
        ind = nanargmin(a[:, 0:n] - o[:, 0:n] + a[:, n:] - o[:, n:], 1)
        sup_inf_diff = 0.5*(a[w, ind] - o[w, ind] + a[w, n+ind] - o[w, n+ind])
        #sup_inf_diff = -sup_inf_diff
        # DEBUG
        #tmp3 = nanmin(a[:, 0:n]-o[:, 0:n]+a[:, n:]-o[:, n:],1)
        #assert all(tmp2==tmp3)
        
        volume = prod(e-y, 1)
        volumeResidual = volume * sup_inf_diff
#        initVolumeResidual = volume * 

    else:
        s, q = o[:, 0:n], o[:, n:2*n]
        Tmp = nanmax(where(q<s, q, s), 1)
#        a_modL, a_modU = a[:, 0:n], a[:, n:2*n]
#        uu = nanmax(where(logical_or(a_modU>a_modL, isnan(a_modU)), a_modU, a_modL), 1)
#        ll = nanmin(where(logical_or(q>s, isnan(q)), s, q), 1)
#        nlhf = log2(uu-ll)
        nlhf = log2(a-o)
#        nlhf = hstack((nlhf[:, :nlhf.shape[1]/2], nlhf[:, nlhf.shape[1]/2:]))

    if p.probType == 'IP':
        F = 0.25 * (a[w, ind] + o[w, ind] + a[w, n+ind] + o[w, n+ind])
        return [si(IP_fields, sup_inf_diff[i], y[i], e[i], o[i], a[i], _s[i], F[i], volume[i], volumeResidual[i]) for i in range(m)]
    else:
        assert p.probType in ('GLP', 'NLP', 'NSP', 'SNLE', 'NLSP')
        return [si(Fields, Tmp[i], y[i], e[i], nlhf[i], nlhc[i] if nlhc is not None else None, o[i], a[i], _s[i]) for i in range(m)]
    
#    else:
#        r18, r19 = r3[:, :n], r3[:, n:]
#        return [si(FuncValFields, Tmp[i], y[i], e[i], nlhf[i], nlhc[i] if nlhc is not None else None, o[i], a[i], _s[i], r18[i], r19[i]) for i in range(m)]

class si:
    def __init__(self, fields, *args, **kwargs):
        for i in range(len(fields)):
            setattr(self, fields[i], args[i])
    
