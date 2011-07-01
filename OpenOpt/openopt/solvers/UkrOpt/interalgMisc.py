from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take, argmax, argmin, abs, hstack, empty, insert, isfinite, append, atleast_2d, \
prod, logical_xor
from interalgLLR import *

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax
    

def r13(y, e, o, a, r3, fTol, varTols, solutions, r6):
    n = r3.shape[1] / 2
    r18, r19 = r3[:, :n], r3[:, n:]
    r5_L, r5_U =  where(logical_and(r18 < fTol, o[:, :n] == 0.0)), where(logical_and(r19 < fTol, o[:, n:] == 0.0))
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
            #r6 = asarray(solutions)
            r6 = append(r6, c.reshape(1, -1), 0)
            
    return solutions, r6

def r14(p, y, e, vv, asdf1, C, CBKPMV, itn, g, nNodes,  \
         frc, fTol, maxSolutions, varTols, solutions, r6, _in, dataType, \
         maxNodes, _s, xRecord):

    isSNLE = p.probType == 'NLSP'
    
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
        _s = atleast_1d(nanmax(a-o))
    y, e, o, a, r3, _s = func7(y, e, o, a, r3, _s)
    
    #y, e = func4(y, e, o, a, fo)

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
        solutions, r6 = r13(y, e, o, a, r3, fTol, varTols, solutions, r6)
        
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


