from numpy import empty, where, logical_and, logical_not, take, logical_or, isnan, zeros, log2, isfinite, \
int8, int16, int32, int64, isinf, asfarray, any, asarray
from interalgLLR import func8, func10
from interalgT import adjustDiscreteVarBounds, truncateByPlane

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax
    
def processConstraints(C0, y, e, _s, p, dataType):
    n = p.n
    m = y.shape[0]
    r15 = empty(m, bool)
    nlh = zeros((m, 2*n))
    r15.fill(True)
    DefiniteRange = True
    
    if len(p._discreteVarsNumList):
        y, e = adjustDiscreteVarBounds(y, e, p)

    for f, r16, r17, tol in C0:
        if p.solver.dataHandling == 'sorted': tol = 0
        
        ip = func10(y, e, p._freeVarsList)
        ip.dictOfFixedFuncs = p.dictOfFixedFuncs
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
        _s = _s[ind]
    return y, e, nlh, None, DefiniteRange, None, _s# indT ; todo: rework it!

def processConstraints2(C0, y, e, _s, p, dataType):
    n = p.n
    m = y.shape[0]
    indT = empty(m, bool)
    indT.fill(False)
    
    for i in range(p.nb):
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, p.A[i], p.b[i])
    for i in range(p.nbeq):
        # TODO: handle it via one func
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, p.Aeq[i], p.beq[i])
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, -p.Aeq[i], -p.beq[i])
   
    
    DefiniteRange = True
    if len(p._discreteVarsNumList):
        adjustDiscreteVarBounds(y, e, p)

    m = y.shape[0]
    nlh = zeros((m, 2*n))
    nlh_0 = zeros(m)
    
    
    for c, f, lb, ub, tol in C0:
        
        m = y.shape[0] # is changed in the cycle
        if m == 0: 
            return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), None, True, False, _s
            #return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), residual.reshape(0, 2*n), True, False, _s
        assert nlh.shape[0] == y.shape[0]
        T0, res, DefiniteRange2 = c.nlh(y, e, p, dataType)
        DefiniteRange = logical_and(DefiniteRange, DefiniteRange2)
        
        assert T0.ndim <= 1, 'unimplemented yet'
        nlh_0 += T0
        assert nlh.shape[0] == m
        # TODO: rework it for case len(p._freeVarsList) >> 1
        if len(res):
            for j, v in enumerate(p._freeVarsList):
                tmp = res.get(v, None)
                if tmp is None:
                    continue
                else:
                    nlh[:, n+j] += tmp[:, tmp.shape[1]/2:].flatten() - T0
                    nlh[:, j] += tmp[:, :tmp.shape[1]/2].flatten() - T0
        assert nlh.shape[0] == m
        ind = where(logical_and(any(isfinite(nlh), 1), isfinite(nlh_0)))[0]
        lj = ind.size
        if lj != m:
            y = take(y, ind, axis=0, out=y[:lj])
            e = take(e, ind, axis=0, out=e[:lj])
            nlh = take(nlh, ind, axis=0, out=nlh[:lj])
            nlh_0 = nlh_0[ind]
    #            residual = take(residual, ind, axis=0, out=residual[:lj])
            indT = indT[ind]
            _s = _s[ind]
            if asarray(DefiniteRange).size != 1: 
                DefiniteRange = take(DefiniteRange, ind, axis=0, out=DefiniteRange[:lj])
        assert nlh.shape[0] == y.shape[0]


        ind = logical_not(isfinite((nlh)))
        if any(ind):
            indT[any(ind, 1)] = True
            
            ind_l,  ind_u = ind[:, :ind.shape[1]/2], ind[:, ind.shape[1]/2:]
            tmp_l, tmp_u = 0.5 * (y[ind_l] + e[ind_l]), 0.5 * (y[ind_u] + e[ind_u])
            y[ind_l], e[ind_u] = tmp_l, tmp_u
            # TODO: mb implement it
            if len(p._discreteVarsNumList):
                if tmp_l.ndim > 1:
                    adjustDiscreteVarBounds(tmp_l, tmp_u, p)
                else:
                    adjustDiscreteVarBounds(y, e, p)

            nlh_l, nlh_u = nlh[:, nlh.shape[1]/2:], nlh[:, :nlh.shape[1]/2]
            
            # copy() is used because += and -= operators are involved on nlh in this cycle and probably some other computations
            nlh_l[ind_u], nlh_u[ind_l] = nlh_u[ind_u].copy(), nlh_l[ind_l].copy()        

    # !! matrix - vector
    nlh += nlh_0.reshape(-1, 1)
        
    residual = None

    return y, e, nlh, residual, DefiniteRange, indT, _s

#def updateNLH(c, y, e, nlh, p):



