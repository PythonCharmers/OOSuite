from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take, argmax, argmin, min, abs, hstack, empty, insert, isfinite, append, atleast_2d, \
prod, logical_xor, argsort, asfarray
from interalgLLR import *

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax

       
#    o = hstack([r[v][0].lb for v in vv] + [r[v][1].lb for v in vv])
#    a = hstack([r[v][0].ub for v in vv] + [r[v][1].ub for v in vv])
#    definiteRange = hstack([r[v][0].definiteRange for v in vv] + [r[v][1].definiteRange for v in vv])
#    # TODO: rework all(definiteRange)
#    return o, a, all(definiteRange)

def r14(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, itn, g, nNodes,  \
         r41, fTol, maxSolutions, varTols, solutions, r6, _in, dataType, \
         maxNodes, _s, indTC, xRecord):

    isSNLE = p.probType in ('NLSP', 'SNLE')
    
    if len(p._discreteVarsNumList):
        adjustDiscreteVarBounds(y, e, p)
    
    Case = p.solver.intervalObtaining

    if Case == 1:
        ip = func10(y, e, vv)
        #o, a = func8(ip, asdf1 + 1e10*p._cons_obj if p._cons_obj is not None else asdf1, dataType)
        o, a, definiteRange = func8(ip, asdf1 + p._cons_obj if p._cons_obj is not None else asdf1, dataType)
    elif Case == 2:
#        o2, a2, definiteRange2 = func82(y, e, vv, asdf1 + p._cons_obj if p._cons_obj is not None else asdf1, dataType)
#        o, a, definiteRange = o2, a2, definiteRange2
        f = asdf1 + p._cons_obj if p._cons_obj is not None else asdf1
        o, a, definiteRange = func82(y, e, vv, f, dataType)
    elif Case == 3:
        # Used for debug
        ip = func10(y, e, vv)
        o, a, definiteRange = func8(ip, asdf1 + p._cons_obj if p._cons_obj is not None else asdf1, dataType)
        
        f = asdf1 + p._cons_obj if p._cons_obj is not None else asdf1
        o2, a2, definiteRange2 = func82(y, e, vv, f, dataType)
        from numpy import allclose
        lf, lf2 = o.copy(), o2.copy()
        lf[isnan(lf)] = 0.123
        lf2[isnan(lf2)] = 0.123
        if not allclose(lf, lf2, atol=1e-10):
            raise 0
        uf, uf2 = a.copy(), a2.copy()
        uf[isnan(uf)] = 0.123
        uf2[isnan(uf2)] = 0.123
        if not allclose(uf, uf2, atol=1e-10):
            raise 0
    
    if p.debug and any(a + 1e-15 < o):  
        p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
    if p.debug and any(logical_xor(isnan(o), isnan(a))):
        p.err('bug in FuncDesigner intervals engine')
    
    m, n = e.shape
    o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T
    
    if asdf1.isUncycled and nlhc is not None and not isSNLE and not p.probType.startswith('MI') and all([v.domain is None for v in p._freeVarsList]):# for SNLE fo = 0
        
        # TODO: 
        # handle constraints with restricted domain and matrix definiteRange
        
        if all(definiteRange):
            # TODO: if o has at least one -inf => prob is unbounded
            tmp1 = o[nlhc==0] 
            if tmp1.size != 0:
                tmp1 = nanmin(tmp1)
                
                ## to prevent roundoff issues ##
                tmp1 += 1e-14*abs(tmp1)
                if tmp1 == 0: tmp1 = 1e-300 
                ######################
                
                r41 = nanmin((r41, tmp1)) 
    else:
        pass
#        r42(o, a)

    fo_prev = float(0 if isSNLE else min((r41, r40 - (fTol if maxSolutions == 1 else 0))))
    
    if itn == 0: 
        # TODO: change for constrained probs
        _s = atleast_1d(inf)
    y, e, o, a, _s, nlhc, residual = func7(y, e, o, a, _s, nlhc, residual)    
    if y.size == 0:
        return _in, g, fo_prev, _s, solutions, r6, xRecord, r41, r40
    
    nodes = func11(y, e, nlhc, indTC, residual, o, a, _s, p)
    
    #y, e = func4(y, e, o, a, fo)
    
    newNLH = 0
    
    if p.solver.dataHandling == 'raw':
       
        if nlhc is not None:
            for i, node in enumerate(nodes): node.tnlhf = node.nlhf + node.nlhc
            #new
            #for i, node in enumerate(nodes): node.tnlhf = node.nlhf + node.residual
        else:
            for i, node in enumerate(nodes): node.tnlhf = node.nlhf # TODO: improve it
            
        an = hstack((nodes, _in))
        
        if newNLH:
            o2 = [node.o for node in an]
            a2 = [node.a for node in an]
            tnlh_fixed = [node.tnlhf for node in an]
            tnlh_fixed_local = asarray(tnlh_fixed[:len(nodes)])
        else:
            if len(_in) != 0:
                o2 = vstack((o, [node.o for node in _in]))
                a2 = vstack((a, [node.a for node in _in]))
            else:
                o2, a2 = o, a
            tnlh_fixed = vstack([node.tnlhf for node in an])
            tnlh_fixed_local = tnlh_fixed[:len(nodes)]

        tmp = a.copy()
        
        tmp[tmp>fo_prev] = fo_prev
        tnlh_curr = tnlh_fixed_local - log2(tmp - o)
        
        # TODO: use it instead of code above
        #tnlh_curr = tnlh_fixed_local - log2(where() - o)
    else:
        tnlh_curr = None
    
    # TODO: don't calculate PointVals for zero-p regions
    PointVals, PointCoords = getr4Values(vv, y, e, tnlh_curr, asdf1, C, p.contol, dataType, p) 

    if PointVals.size != 0:
        xk, Min = r2(PointVals, PointCoords, dataType)
    else: # all points have been removed by func7
        xk = p.xk
        Min = nan
    
    if r40 > Min:
        r40 = Min
        xRecord = xk.copy()# TODO: is copy required?
    if r41 > Min:
        r41 = Min
    
    fo = float(0 if isSNLE else min((r41, r40 - (fTol if maxSolutions == 1 else 0))))
        
    if p.solver.dataHandling == 'raw':
        # TODO: check it with bool/integer variables
        tnlh_curr = []
        r10 = []
        NN = []
        if newNLH:
            for i in range(len(an)):
                # TODO: optimize it
                tmp = a2[i].copy()
                tmp[tmp>fo] = fo
                if fo == inf and any(tmp==inf):
                    tmp[tmp==inf] = o2[tmp==inf]
                if nanmax(tmp - o2) < 0:
                    r10.append(i)
                TMP = tnlh_fixed[i] - log2(tmp - o2[i])
                tnlh_curr.append(TMP)
                NN.append(nanmin(TMP))
            NN = asarray(NN)
            r10 = asarray(r10)
            NN = asarray(NN)
        else:
            tmp = a2.copy()
            tmp[tmp>fo] = fo
            ind = tmp==inf
            if any(ind):
                tmp[ind] = o2[ind] + 1.0 # TODO: rework it
            #tnlh_curr = tnlh_fixed - log2(fo - o2)
            tnlh_curr = tnlh_fixed - log2(tmp - o2)
            r10 = where(nanmax(tmp-o2, 1) < 0)[0]
            #r10 = where(o2>=fo)[0]
            # TODO: optimize it, don't recalculate for whole stored arrays
            #ind_u_inf = where(a2==inf)[0]
            
        if r10.size != 0:
            mino = [an[i].o for i in r10]
            mmlf = nanmin(asarray(mino))
            g = min((g, mmlf))

#        if ind_u_inf.size != 0:
#            S = 1.0 # TODO: set better value
#            tnlh_curr[ind_u_inf] = S+(asfarray([an[i].nlhc for i in ind_u_inf]) if len(C) != 0 else 0)
        
        for i, node in enumerate(an): node.tnlh_curr = tnlh_curr[i]
        
        if not newNLH:
            NN = nanmin(tnlh_curr, 1)
        r10 = logical_or(isnan(NN), NN == inf)
        
        if any(r10):
            ind = where(logical_not(r10))[0]
            an = take(an, ind, axis=0, out=an[:ind.size])
            #tnlh = take(tnlh, ind, axis=0, out=tnlh[:ind.size])
            NN = take(NN, ind, axis=0, out=NN[:ind.size])
        
        if not isSNLE or p.maxSolutions == 1:
            astnlh = argsort(NN)
            an = an[astnlh]
        # NEW END   
    
    # OLD
    if p.solver.dataHandling == 'sorted':
        if isSNLE and p.maxSolutions != 1: 
            an = hstack((nodes, _in))
        else:
            nodes.sort(key = lambda obj: obj.key)

            if len(_in) == 0:
                an = nodes
            else:
                arr1 = [node.key for node in _in]
                arr2 = [node.key for node in nodes]
                r10 = searchsorted(arr1, arr2)
                an = insert(_in, r10, nodes)

        
    if maxSolutions != 1:
        solutions, r6 = r13(o, a, PointCoords, PointVals, fTol, varTols, solutions, r6)
        
        p._nObtainedSolutions = len(solutions)
        if p._nObtainedSolutions > maxSolutions:
            solutions = solutions[:maxSolutions]
            p.istop = 0
            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
            return an, g, fo, None, solutions, r6, xRecord, r41, r40
    
    #p.iterfcn(xk, Min)
    p.iterfcn(xRecord, r40)
    if p.istop != 0: 
        return an, g, fo, None, solutions, r6, xRecord, r41, r40
    if isSNLE and maxSolutions == 1 and Min <= fTol:
        # TODO: rework it for nonlinear systems with non-bound constraints
        p.istop, p.msg = 1000, 'required solution has been obtained'
        return an, g, fo, None, solutions, r6, xRecord, r41, r40
    
    
    an, g = func9(an, fo, g, p)

    nn = maxNodes#1 if asdf1.isUncycled and all(isfinite(o)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes
    
    an, g = func5(an, nn, g, p)
    nNodes.append(len(an))
    return an, g, fo, _s, solutions, r6, xRecord, r41, r40


def r13(o, a, PointCoords, PointVals, fTol, varTols, solutions, r6):
    n = o.shape[1] / 2
    
    #L1, L2 = o[:, :n], o[:, n:]
    #omin = where(logical_or(L1 > L2, isnan(L1)), L2, L1)
    #r5Ind =  where(logical_and(PointVals < fTol, nanmax(omin, 1) == 0.0))[0]
    
    r5Ind =  where(PointVals < fTol)[0]

    r5 = PointCoords[r5Ind]
    
    for c in r5:
        if len(solutions) == 0 or not any(all(abs(c - r6) < varTols, 1)): 
            solutions.append(c)
            #r6 = asarray(solutions)
            r6 = append(r6, c.reshape(1, -1), 0)
            
    return solutions, r6
