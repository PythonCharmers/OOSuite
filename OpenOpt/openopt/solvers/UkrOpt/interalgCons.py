from numpy import empty, where, logical_and, take, logical_or, isnan, zeros, log2, isfinite, int8, int16, int32, int64, inf, isinf
from interalgLLR import func8

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    
def processConstraints(C, y, e, ip, m, p, dataType):
    n = p.n
    r15 = empty(m, bool)
    nlh = zeros((m, 2*n))
    r15.fill(True)

    # here tol is unused 
    for i, (f, r16, r17, tol) in enumerate(C):
        o, a = func8(ip, f, dataType)
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
        if r16 == r17:
            # TODO: for non-exact interval quality increase nlh while moving from 0.5*(e-y)
            nlh[r16 > a] = inf
            nlh[r17 < o] = inf
            #nlh += log2(aor20)# TODO: mb use other
        elif isfinite(r16) and not isfinite(r17):
            tmp = (r16 - o) / aor20
            
            tmp[r16 > a] = 0
            tmp[isinf(a)] = 1
            tmp[r16 <= o] = 1
            
            nlh -= log2(tmp)
        elif isfinite(r17) and not isfinite(r16):
            tmp = (a - r17) / aor20
            
            tmp[r17 < o] = 0
            tmp[isinf(o)] = 1
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
    return y, e, nlh
