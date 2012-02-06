from numpy import isnan, take, any, all, logical_or, logical_and, logical_not, atleast_1d, where, \
asarray, inf, nan, argmin, argsort, tile, searchsorted
from bisect import bisect_right
from FuncDesigner.Interval import adjust_lx_WithDiscreteDomain, adjust_ux_WithDiscreteDomain
try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax

def r42(o, a):
#    n_where_lx_2 = where(y <=-0.5)[0].size
#    n_where_ux_2 = where(e >=-0.5)[0].size
#    nn = n_where_lx_2 + n_where_ux_2
    m, N = o.shape
    n = N / 2
    o_l, o_u = o[:, :n], o[:, n:]
    a_l, a_u = a[:, :n], a[:, n:]
    o_m = where(logical_or(o_l < o_u, isnan(o_u)), o_l, o_u)
    a_m = where(logical_or(a_l < a_u, isnan(a_l)), a_u, a_l)
    o_M = nanmax(o_m, 1)
    a_M = nanmin(a_m, 1)
    # TODO: make it matrix-vector componentwise
    o_M = tile(o_M.reshape(m, 1), (1, 2*n))
    ind = o < o_M
    if any(ind):
        o[ind] = o_M[ind]
    a_M = tile(a_M.reshape(m, 1), (1, 2*n))        
    ind = a > a_M
    if any(ind):
        a[ind] = a_M[ind]

#    n_where_lx_2 = where(y <=-0.5)[0].size
#    n_where_ux_2 = where(e >=-0.5)[0].size
#    nn2 = n_where_lx_2 + n_where_ux_2
#    print nn, nn2
#    assert nn == nn2


def adjustDiscreteVarBounds(y, e, p):
    n = p.n
    # TODO: remove the cycle, use vectorization
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        
#        n_where_lx_2 = where(y[:, i] <=-0.5)[0].size
#        n_where_ux_2 = where(e[:, i] >=-0.5)[0].size
#        nn = n_where_lx_2 + n_where_ux_2
        
        adjust_lx_WithDiscreteDomain(y[:, i], v)
        adjust_ux_WithDiscreteDomain(e[:, i], v)
        
#        n_where_lx_2 = where(y[:, i] <=-0.5)[0].size
#        n_where_ux_2 = where(e[:, i] >=-0.5)[0].size
#        nn2 = n_where_lx_2 + n_where_ux_2
#        print nn, nn2, p.iter
#        if nn != nn2:
#            raise 0

#        ind = searchsorted(v.domain, y[:, i], 'left')
#        ind2 = searchsorted(v.domain, y[:, i], 'right')
#        ind3 = where(ind!=ind2)[0]
#        #Tmp = y[:, ind3].copy()
#        Tmp = v.domain[ind[ind3]]
#        ind[ind==v.domain.size] -= 1
#        ind[ind==v.domain.size-1] -= 1
#        y[:, i] = v.domain[ind+1]
#        y[:, i][ind3] = Tmp
#        
#        
#        ind = searchsorted(v.domain, e[:, i], 'left')
#        ind2 = searchsorted(v.domain, e[:, i], 'right')
#        ind3 = where(ind!=ind2)[0]
#        #Tmp = e[:, ind3].copy()
#        Tmp = v.domain[ind[ind3]]
#        #ind[ind==v.domain.size] -= 1
#        ind[ind==0] = 1
#        e[:, i] = v.domain[ind-1]
#        e[:, i][ind3] = Tmp

    ind = any(y>e, 1)
    if any(ind):
        ind = where(logical_not(ind))[0]
        s = ind.size
        y = take(y, ind, axis=0, out=y[:s])
        e = take(e, ind, axis=0, out=e[:s])
    


def func7(y, e, o, a, _s, nlhc, residual):
    r10 = logical_and(all(isnan(o), 1), all(isnan(a), 1))
    if any(r10):
        j = where(logical_not(r10))[0]
        lj = j.size
        y = take(y, j, axis=0, out=y[:lj])
        e = take(e, j, axis=0, out=e[:lj])
        o = take(o, j, axis=0, out=o[:lj])
        a = take(a, j, axis=0, out=a[:lj])
        _s = _s[j]
        if nlhc is not None:
            nlhc = take(nlhc, j, axis=0, out=nlhc[:lj])
        if residual is not None:
            residual = take(residual, j, axis=0, out=residual[:lj])
    return y, e, o, a, _s, nlhc, residual

def func9(an, fo, g, p):
    
    #ind = searchsorted(ar, fo, side='right')
    if p.probType in ('NLSP', 'SNLE') and p.maxSolutions != 1:
        mino = atleast_1d([node.key for node in an])
        ind = mino > 0
        if not any(ind):
            return an, g
        else:
            g = nanmin((g, nanmin(mino[ind])))
            ind2 = where(logical_not(ind))[0]
            an = take(an, ind2, axis=0, out=an[:ind2.size])
            return an, g
            
        
    elif p.solver.dataHandling == 'sorted':
        #OLD
        mino = [node.key for node in an]
        ind = bisect_right(mino, fo)
        if ind == len(mino):
            return an, g
        else:
            g = nanmin((g, nanmin(atleast_1d(mino[ind]))))
            return an[:ind], g
    elif p.solver.dataHandling == 'raw':
        
        #NEW
        mino = [node.key for node in an]
        mino = atleast_1d(mino)
        r10 = mino > fo
        if not any(r10):
            return an, g
        else:
            ind = where(r10)[0]
            g = nanmin((g, nanmin(atleast_1d(mino)[ind])))
            an = asarray(an)
            ind2 = where(logical_not(r10))[0]
            an = take(an, ind2, axis=0, out=an[:ind2.size])
            return an, g

        # NEW 2
#        curr_tnlh = [node.tnlh_curr for node in an]
#        import warnings
#        warnings.warn('! fix g')
        
        return an, g
        
    else:
        assert 0, 'incorrect nodes remove approach'

def func5(an, nn, g, p):
    m = len(an)
    if m <= nn: return an, g
    
    mino = [node.key for node in an]
    
    if nn == 1: # box-bound probs with exact interval analysis
        ind = argmin(mino)
        assert ind in (0, 1), 'error in interalg engine'
        g = nanmin((mino[1-ind], g))
        an = atleast_1d([an[ind]])
    elif m > nn:
        if p.solver.dataHandling == 'raw':
            ind = argsort(mino)
            th = mino[ind[nn]]
            ind2 = where(mino < th)[0]
            g = nanmin((th, g))
            an = take(an, ind2, axis=0, out=an[:ind2.size])
        else:
            g = nanmin((mino[nn], g))
            an = an[:nn]
    return an, g

def func4(y, e, o, a, fo):
    if fo is None: return # used in IP
    cs = (y + e)/2
    n = y.shape[1]
    s, q = o[:, 0:n], o[:, n:2*n]
    ind = logical_or(s > fo, isnan(s)) # TODO: assert isnan(s) is same to isnan(a_modL)
    indT = any(ind, 1)
    if any(ind):
        y[ind] = cs[ind]
        # Changes
#        ind = logical_and(ind, logical_not(isnan(a[:, n:2*n])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii
#        a[:, 0:n][ind] = a[:, n:2*n][ind]
#        o[:, 0:n][ind] = o[:, n:2*n][ind]
    ind = logical_or(q > fo, isnan(q)) # TODO: assert isnan(q) is same to isnan(a_modU)
    indT = logical_or(any(ind, 1), indT)
    if any(ind):
        # copy is used to prevent y and e being same array, that may be buggy with discret vars
        e[ind] = cs[ind].copy() 
        # Changes
#        ind = logical_and(ind, logical_not(isnan(a[:, n:])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii
#        a[:, n:2*n][ind] = a[:, 0:n][ind]
#        o[:, n:2*n][ind] = o[:, 0:n][ind]
    return y, e, indT

