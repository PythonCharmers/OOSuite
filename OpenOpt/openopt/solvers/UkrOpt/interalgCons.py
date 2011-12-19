from numpy import empty, where, logical_and, take, logical_or, isnan, zeros, log2, isfinite, int8, int16, int32, int64, inf, isinf, asfarray
from interalgLLR import func8

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    
def processConstraints(C0, y, e, ip, m, p, dataType):
    n = p.n
    r15 = empty(m, bool)
    nlh = zeros((m, 2*n))
    r15.fill(True)
    DefiniteRange = True
    for i, (f, r16, r17, tol) in enumerate(C0):
        if p.solver.dataHandling == 'sorted': tol = 0
        o, a, definiteRange = func8(ip, f, dataType)
        DefiniteRange = logical_and(DefiniteRange, definiteRange)
        m = o.size/(2*n)
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
            tmp[r24 == 0.0] = 1.0 # may be encountered if a == o, especially for integer probs
            tmp[tmp<1e-300] = 1e-300 # TODO: improve it
            nlh -= log2(tmp)
            #nlh += log2(aor20)# TODO: mb use other
            #nlh -= log2(a-r17) + log2(r16-o)
            #nlh += log2((a-r17)/ aor20) + log2((r16-o)/ aor20)
            # TODO: for non-exact interval quality increase nlh while moving from 0.5*(e-y)
            nlh[val > a] = inf
            nlh[val < o] = inf
        elif isfinite(r16) and not isfinite(r17):
            #OLD
            tmp = (r16 - o + tol) / aor20
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
            tmp[isinf(a)] = 1 # (to prevent inf/inf=nan); TODO: rework it
            #tmp[r16 - tol <= o] = 1
            tmp[r16 <= o] = 1
            
            nlh -= log2(tmp)
        elif isfinite(r17) and not isfinite(r16):
            #OLD
            tmp = (a - r17+tol) / aor20
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
#            from numpy import ones_like
#            tmp = 0.5*ones_like(tmp)
            
            tmp[r17 < o] = 0
            tmp[isinf(o)] = 1 # (to prevent inf/inf=nan);TODO: rework it
            #tmp[r17+tol >= a] = 1
            tmp[r17 >= a] = 1
            
            nlh -= log2(tmp) 
        else:
            p.err('this part of interalg code is unimplemented for double-box-bound constraints yet')
        
#                ind = logical_or(o_ > r17 + p.contol, isnan(o_)) 
#                if any(ind):
#                    r4_L_ind = logical_or(r4_L_ind, ind)
#                    
#                ind = logical_or(a_ < r16 - p.contol, isnan(a_)) 
#                if any(ind):
#                    r4_U_ind = logical_or(r4_U_ind, ind)
#            
#            cs = 0.5*(y + e)
#            ind = r4_L_ind
#            if any(ind):
#                y[ind] = cs[ind]
#            ind = r4_U_ind
#            if any(ind):
#                e[ind] = cs[ind]
        
    ind = where(r15)[0]
    lj = ind.size
    y = take(y, ind, axis=0, out=y[:lj])
    e = take(e, ind, axis=0, out=e[:lj])
    nlh = take(nlh, ind, axis=0, out=nlh[:lj])
    return y, e, nlh, DefiniteRange
