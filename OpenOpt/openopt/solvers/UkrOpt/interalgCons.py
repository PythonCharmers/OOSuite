from numpy import empty, where, logical_and, logical_not, take, logical_or, isnan, zeros, log2, isfinite, \
int8, int16, int32, int64, inf, isinf, asfarray, hstack, vstack, prod, all, any, asarray, tile
from interalgLLR import func8, func82, func10

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    
def processConstraints(C0, y, e, p, dataType):
    n = p.n
    m = y.shape[0]
    r15 = empty(m, bool)
    #NEW = p.solver.intervalObtaining == 2
    nlh = zeros((m, 2*n))
    r15.fill(True)
    DefiniteRange = True
  
    for f, r16, r17, tol in C0:
        if p.solver.dataHandling == 'sorted': tol = 0
        
        ip = func10(y, e, p._freeVarsList)
        o, a, definiteRange = func8(ip, f, dataType)
            
        DefiniteRange = logical_and(DefiniteRange, definiteRange)
        o, a  = o.reshape(2*n, m).T, a.reshape(2*n, m).T
        lf1, lf2, uf1, uf2 = o[:, 0:n], o[:, n:2*n], a[:, 0:n], a[:, n:2*n]
        o_ = where(logical_or(lf1>lf2, isnan(lf1)), lf2, lf1)
        a_ = where(logical_or(uf1>uf2, isnan(uf2)), uf1, uf2)
        om, am = nanmin(o_, 1), nanmax(a_, 1)
        
        ind = logical_and(am >= r16, om  <= r17)
        r15 = logical_and(r15, ind)
        aor20 = a - o
        if dataType in [int8, int16, int32, int64, int]:
            aor20 = asfarray(aor20)
        #aor20[aor20 > 1e200] = 1e200
        a_t,  o_t = a.copy(), o.copy()
        if dataType in [int8, int16, int32, int64, int]:
            a_t,  o_t = asfarray(a_t), asfarray(o_t)
        if r16 == r17:
            val = r17
            a_t[a_t > val + tol] = val + tol
            o_t[o_t < val - tol] = val - tol
            r24 = a_t - o_t
            tmp = r24 / aor20
            tmp[logical_or(isinf(o), isinf(a))] = 1e-10 #  (to prevent inf/inf=nan); TODO: rework it
            
            tmp[r24 == 0.0] = 1.0 # may be encountered if a == o, especially for integer probs
            tmp[tmp<1e-300] = 1e-300 # TODO: improve it
            #nlh += log2(aor20)# TODO: mb use other
            #nlh -= log2(a-r17) + log2(r16-o)
            #nlh += log2((a-r17)/ aor20) + log2((r16-o)/ aor20)
            # TODO: for non-exact interval quality increase nlh while moving from 0.5*(e-y)
            tmp[val > a] = 0
            tmp[val < o] = 0
        elif isfinite(r16) and not isfinite(r17):
            tmp = (a - r16 + tol) / aor20
            
            tmp[logical_and(isinf(o), logical_not(isinf(a)))] = 1e-10 # (to prevent inf/inf=nan); TODO: rework it
            tmp[isinf(a)] = 1-1e-10 # (to prevent inf/inf=nan); TODO: rework it
            
            #tmp = (a - r16) / aor20
            #NEW
#            o_t[o_t < r16 - tol] = r16 - tol
#            #ind = a_t>o
#            #a_t[ind] = o[ind]
#            #o_t[o_t < r16 - tol] = r16 - tol
#            r24 = a - o_t
#            tmp = r24 / aor20
            #tmp = (a - r16) / aor20f
            
            tmp[tmp<1e-300] = 1e-300 # TODO: improve it
            tmp[tmp>1.0] = 1.0
            
            tmp[r16 > a] = 0
            
            #tmp[r16 - tol <= o] = 1
            tmp[r16 <= o] = 1

        elif isfinite(r17) and not isfinite(r16):
            tmp = (r17-a+tol) / aor20
            
            tmp[isinf(o)] = 1-1e-10 # (to prevent inf/inf=nan);TODO: rework it
            tmp[logical_and(isinf(a), logical_not(isinf(o)))] = 1e-10 # (to prevent inf/inf=nan); TODO: rework it
            
            #tmp = (r17-o) / aor20
            #NEW
#            a_t[a_t > r17 + tol] = r17 + tol
#            
#            #r24 = a - r17
#            r24  = a_t - o
#            #r24[r24<0] = 0.0
#            tmp = r24 / aor20
            
            tmp[tmp<1e-300] = 1e-300 # TODO: improve it
            tmp[tmp>1.0] = 1.0
            
            tmp[r17 < o] = 0
            
            #tmp[r17+tol >= a] = 1
            tmp[r17 >= a] = 1

        else:
            p.err('this part of interalg code is unimplemented for double-box-bound constraints yet')
            
        nlh -= log2(tmp) 
        
    ind = where(r15)[0]
    lj = ind.size
    if lj != m:
        y = take(y, ind, axis=0, out=y[:lj])
        e = take(e, ind, axis=0, out=e[:lj])
        nlh = take(nlh, ind, axis=0, out=nlh[:lj])
    return y, e, nlh, DefiniteRange

def processConstraints2(C0, y, e, p, dataType):
    n = p.n
    m = y.shape[0]
#    r15 = empty(m, bool)
#    r15.fill(True)
#    r152 = {}
    nlh = zeros((m, 2*n))
    #nlh2 = {} 
    
    DefiniteRange = True
    
    for f, r16, r17, tol in C0:
        
        m = y.shape[0] # is changed in the cycle
        if m == 0: return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), True
        
        if p.solver.dataHandling == 'sorted': tol = 0
        #o, a, definiteRange = func82(y, e, p._freeVarsList, f, dataType)
        
        # TODO: use cut each cycle turn
        domain = dict([(v, (y[:, k], e[:, k])) for k, v in enumerate(p._freeVarsList)])
        
        r, r0 = f.iqg(domain, dataType)
        dep = f._getDep() # TODO: Rework it for fixed vars
        isSubset = len(dep) < n
        
        
        if isSubset:
            o, a = r0.lb, r0.ub
            
            # using tile to make shape like it was divided into 2 boxes
            # todo: optimize it
            tmp = getTmp(tile(o, (2, 1)), tile(a, (2, 1)), r16, r17, tol, m, dataType)
            T0 = log2(tmp[:, tmp.shape[1]/2:])
        
        for j, v in enumerate(p._freeVarsList):
            if v in dep:
                o, a = vstack((r[v][0].lb, r[v][1].lb)), vstack((r[v][0].ub, r[v][1].ub))
                
                # TODO: 1) FIX IT it for matrix definiteRange
                # 2) seems like DefiniteRange = (True, True) for any variable is enough for whole range to be defined in the involved node
                DefiniteRange = logical_and(DefiniteRange, r[v][0].definiteRange)
                DefiniteRange = logical_and(DefiniteRange, r[v][1].definiteRange)
                
                tmp = log2(getTmp(o, a, r16, r17, tol, m, dataType))
               
                nlh[:, n+j] -= tmp[:, tmp.shape[1]/2:].flatten()
                nlh[:, j] -= tmp[:, :tmp.shape[1]/2].flatten()
            else:
                nlh[:, j] -= T0.flatten()
                nlh[:, n+j] -= T0.flatten()

        ind = where(any(isfinite(nlh), 1))[0]
        
        lj = ind.size
        if lj != m:
            y = take(y, ind, axis=0, out=y[:lj])
            e = take(e, ind, axis=0, out=e[:lj])
            nlh = take(nlh, ind, axis=0, out=nlh[:lj])
            if asarray(DefiniteRange).size != 1: 
                DefiniteRange = take(DefiniteRange, ind, axis=0, out=DefiniteRange[:lj])
                
        ind = logical_not(isfinite((nlh)))
        if any(ind):
            ind_l,  ind_u = ind[:, :ind.shape[1]/2], ind[:, ind.shape[1]/2:]
            y[ind_l], e[ind_u] = 0.5 * (y[ind_l] + e[ind_l]), 0.5 * (y[ind_u] + e[ind_u])
            
            nlh_l, nlh_u = nlh[:, nlh.shape[1]/2:], nlh[:, :nlh.shape[1]/2]
            
            # copy() is used bacause += and -= operators are involved on nlh in this cycle and probably some other computations
            nlh_l[ind_u], nlh_u[ind_l] = nlh_u[ind_u].copy(), nlh_l[ind_l].copy()
            
#    print nlh
#    from numpy import diff
#    print diff(nlh)
    return y, e, nlh, DefiniteRange

def getTmp(o, a, r16, r17, tol, m, dataType):
   
    #print o.shape
    M = prod(o.shape) / (2*m)
    #init
    o, a  = o.reshape(2*M, m).T, a.reshape(2*M, m).T
    #1
    #o, a  = o.reshape(m, 2*M), a.reshape(m, 2*M)
    
    lf1, lf2, uf1, uf2 = o[:, 0:M], o[:, M:2*M], a[:, 0:M], a[:, M:2*M]
    o_ = where(logical_or(lf1>lf2, isnan(lf1)), lf2, lf1)
    a_ = where(logical_or(uf1>uf2, isnan(uf2)), uf1, uf2)
    om, am = nanmin(o_, 1), nanmax(a_, 1)
    
    ind = logical_and(am >= r16, om  <= r17)
        
    aor20 = a - o
    if dataType in [int8, int16, int32, int64, int]:
        aor20 = asfarray(aor20)
    #aor20[aor20 > 1e200] = 1e200
    a_t,  o_t = a.copy(), o.copy()
    if dataType in [int8, int16, int32, int64, int]:
        a_t,  o_t = asfarray(a_t), asfarray(o_t)
    if r16 == r17:
        val = r17
        a_t[a_t > val + tol] = val + tol
        o_t[o_t < val - tol] = val - tol
        r24 = a_t - o_t
        tmp = r24 / aor20
        tmp[logical_or(isinf(o), isinf(a))] = 1e-10 #  (to prevent inf/inf=nan); TODO: rework it
        
        tmp[r24 == 0.0] = 1.0 # may be encountered if a == o, especially for integer probs
        tmp[tmp<1e-300] = 1e-300 # TODO: improve it

        # TODO: for non-exact interval quality increase nlh while moving from 0.5*(e-y)
        tmp[val > a] = 0
        tmp[val < o] = 0
    elif isfinite(r16) and not isfinite(r17):
        tmp = (a - r16 + tol) / aor20
        
        tmp[logical_and(isinf(o), logical_not(isinf(a)))] = 1e-10 # (to prevent inf/inf=nan); TODO: rework it
        tmp[isinf(a)] = 1-1e-10 # (to prevent inf/inf=nan); TODO: rework it
        
        tmp[tmp<1e-300] = 1e-300 # TODO: improve it
        tmp[tmp>1.0] = 1.0
        
        tmp[r16 > a] = 0
        
        #tmp[r16 - tol <= o] = 1
        tmp[r16 <= o] = 1
        
    elif isfinite(r17) and not isfinite(r16):
        tmp = (r17-o+tol) / aor20
        #tmp = (r17-a+tol) / aor20
        
        tmp[isinf(o)] = 1-1e-10 # (to prevent inf/inf=nan);TODO: rework it
        tmp[logical_and(isinf(a), logical_not(isinf(o)))] = 1e-10 # (to prevent inf/inf=nan); TODO: rework it
        
        tmp[tmp<1e-300] = 1e-300 # TODO: improve it
        tmp[tmp>1.0] = 1.0
        
        tmp[r17 < o] = 0
        
        #tmp[r17+tol >= a] = 1
        tmp[r17 >= a] = 1

    else:
        p.err('this part of interalg code is unimplemented for double-box-bound constraints yet')
        
    return tmp
