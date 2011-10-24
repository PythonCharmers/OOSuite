from numpy import isnan, take, any, all, logical_or, logical_and, logical_not, atleast_1d, where, asarray, inf, nan, argmin, argsort
from bisect import bisect_right
try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    
def func7(y, e, o, a, _s):
    r10 = logical_and(all(isnan(o), 1), all(isnan(a), 1))
    if any(r10):
        j = where(logical_not(r10))[0]
        lj = j.size
        y = take(y, j, axis=0, out=y[:lj])
        e = take(e, j, axis=0, out=e[:lj])
        o = take(o, j, axis=0, out=o[:lj])
        a = take(a, j, axis=0, out=a[:lj])
        _s = _s[j]
    return y, e, o, a, _s

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

def func4(y, e, o, a, nlhc, fo):
    if fo is None: return # used in IP
    cs = (y + e)/2
    n = y.shape[1]
    s, q = o[:, 0:n], o[:, n:2*n]
    if nlhc[0] is not None:
        nlhc_modL, nlhc_modU = nlhc[:, 0:n], nlhc[:, n:2*n]
    ind = logical_or(s > fo, isnan(s)) # TODO: assert isnan(s) is same to isnan(a_modL)
    if nlhc[0] is not None:
        ind = logical_or(ind, logical_or(isnan(nlhc_modL), nlhc_modL == inf))
    if any(ind):
        y[ind] = cs[ind]
        # Changes
#        ind = logical_and(ind, logical_not(isnan(a[:, n:2*n])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii
#        a[:, 0:n][ind] = a[:, n:2*n][ind]
#        o[:, 0:n][ind] = o[:, n:2*n][ind]
    ind = logical_or(q > fo, isnan(q)) # TODO: assert isnan(q) is same to isnan(a_modU)
    if nlhc[0] is not None:
        ind = logical_or(ind, logical_or(isnan(nlhc_modU), nlhc_modU == inf))
    if any(ind):
        e[ind] = cs[ind]
        # Changes
#        ind = logical_and(ind, logical_not(isnan(a[:, n:])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii
#        a[:, n:2*n][ind] = a[:, 0:n][ind]
#        o[:, n:2*n][ind] = o[:, 0:n][ind]
    return y, e