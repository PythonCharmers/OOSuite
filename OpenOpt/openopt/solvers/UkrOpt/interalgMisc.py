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

def r14(p, nlhc, definiteRange, y, e, vv, asdf1, C, r40, itn, g, nNodes,  \
         r41, fTol, maxSolutions, varTols, solutions, r6, _in, dataType, \
         maxNodes, _s, xRecord):

    isSNLE = p.probType in ('NLSP', 'SNLE')
    
    
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
        

#        o = hstack([r[v][0].lb for v in vv] + [r[v][1].lb for v in vv])
#        a = hstack([r[v][0].ub for v in vv] + [r[v][1].ub for v in vv])
        #definiteRange = hstack([r[v][0].definiteRange for v in vv] + [r[v][1].definiteRange for v in vv])
        # TODO: rework all(definiteRange)
        #return o, a, definiteRange#all(definiteRange)

#    else:
#        ip = func10(y, e, vv)
#        #o, a = func8(ip, asdf1 + 1e10*p._cons_obj if p._cons_obj is not None else asdf1, dataType)
#        o, a, definiteRange = func8(ip, asdf1 + p._cons_obj if p._cons_obj is not None else asdf1, dataType)
#        o2, a2, definiteRange2 = func82(y, e, vv, asdf1 + p._cons_obj if p._cons_obj is not None else asdf1, dataType)
#        print 'diff:',  max(abs(o-o2)), max(abs(a-a2))

    #ss = o2.size/2
    #print (o2, o, )
    #print max(abs(o2[:ss]-o[:ss])), max(abs(a2[:ss]-a[:ss])), max(abs(o2[ss:]-o[ss:])), max(abs(a2[ss:]-a[ss:]))
    #print max(abs(o2-o)), max(abs(a2-a))
#        print o.shape, o2.shape, definiteRange.shape, definiteRange2.shape
    
    if p.debug and any(a + 1e-15 < o):  
        p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
    if p.debug and any(logical_xor(isnan(o), isnan(a))):
        p.err('bug in FuncDesigner intervals engine')
    
    m, n = e.shape
    o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T
    
#    if asdf1.isUncycled and nlhc is not None and not isSNLE:# for SNLE fo = 0
#        
#        # TODO: 
#        # handle constraints with restricted domain and matrix definiteRange
#        
#        if all(definiteRange):
#            tmp1 = o[nlhc==0]
#            if tmp1.size != 0:
#                tmp1 = nanmin(tmp1)
#                tmp1 += 1e-5*abs(tmp1)
#                print tmp1
#                r41 = nanmin((r41, tmp1)) 

    
    fo_prev = float(0 if isSNLE else min((r41, r40 - (fTol if maxSolutions == 1 else 0))))
    
    
    if itn == 0: 
        # TODO: change for constrained probs
        _s = atleast_1d(nanmax(a-o))
    y, e, o, a, _s = func7(y, e, o, a, _s)    
    if y.size == 0:
        return _in, g, fo_prev, _s, solutions, r6, xRecord, r41, r40
    
    nodes = func11(y, e, nlhc, o, a, _s, p)
    
    #y, e = func4(y, e, o, a, fo)
    
    if p.solver.dataHandling == 'raw':
        # NEW
        if len(_in) != 0:
            o2 = vstack((o, [node.o for node in _in]))
            a2 = vstack((a, [node.a for node in _in]))
        else:
            o2, a2 = o, a
        
#        nlhf_fixed = log2(a2-o2)
#        if nlhc is None:
#            for i, node in enumerate(nodes): node.tnlhf = nlhf_fixed[i]
#        else:
#            for i, node in enumerate(nodes): node.tnlhf = nlhf_fixed[i] + nlhc[i]
        
        if nlhc is not None:
            #nlhf_fixed = log2(a2-o2)
            for i, node in enumerate(nodes): node.tnlhf = node.nlhf + node.nlhc
        else:
            for i, node in enumerate(nodes): node.tnlhf = node.nlhf # TODO: improve it
            
        an = hstack((nodes, _in))
        tnlh_fixed = vstack([node.tnlhf for node in an])
    
    
        #NEW
        tnlh_fixed_local = tnlh_fixed[:len(nodes)]
        
        tmp = a.copy()
        
        tmp[tmp>fo_prev] = fo_prev
        #tnlh_curr = tnlh_fixed - log2(fo - o2)
        tnlh_curr = tnlh_fixed_local - log2(tmp - o)
        
        # TODO: use it instead of code above
        #tnlh_curr = tnlh_fixed_local - log2(where() - o)
    else:
        tnlh_curr = None
    
    # TODO: don't calculate PointVals for zero-p regions
    PointVals, PointCoords = getr4Values(vv, y, e, tnlh_curr, asdf1, C, p.contol, dataType, p) 

    if 1 or PointVals.size != 0:
        xk, Min = r2(PointVals, PointCoords, dataType)
    else:# all points have been removed by func7
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
        tmp = a2.copy()
        tmp[tmp>fo] = fo
        #tnlh_curr = tnlh_fixed - log2(fo - o2)
        tnlh_curr = tnlh_fixed - log2(tmp - o2)
        
        r10 = where(nanmax(tmp - o2, 1) < 0)
        if any(r10):
            mino = [node.key for node in an]
            mmlf = nanmin(asarray(mino)[r10])
            g = min((g, mmlf))
        
        # TODO: optimize it, don't recalculate for whole stored arrays
        ind = where(a2==inf)[0]
        if ind.size != 0:
            S = 1.0 # TODO: set better value
            tnlh_curr[ind] = S+(asfarray([an[i].nlhc for i in ind]) if len(C) != 0 else 0)
        
        for i, node in enumerate(an): node.tnlh_curr = tnlh_curr[i]
        
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

    nn = 1 if asdf1.isUncycled and all(isfinite(o)) and p._isOnlyBoxBounded else maxNodes
    
    an, g = func5(an, nn, g, p)
    nNodes.append(len(an))
    return an, g, fo, _s, solutions, r6, xRecord, r41, r40


def r13(o, a, PointCoords, PointVals, fTol, varTols, solutions, r6):
    n = o.shape[1] / 2
    
    #L1, L2 = o[:, :n], o[:, n:]
    #omin = where(logical_or(L1 > L2, isnan(L1)), L2, L1)
    #r5Ind =  where(logical_and(PointVals < fTol, nanmax(omin, 1) == 0.0))[0]
    
    r5Ind =  where(PointVals < fTol)[0]
    
    r5 = []
    for i in r5Ind:#TODO: rework it
        r5.append(PointCoords[i])
    
    for c in r5:
        if len(solutions) == 0 or not any(all(abs(c - r6) < varTols, 1)): 
            solutions.append(c)
            #r6 = asarray(solutions)
            r6 = append(r6, c.reshape(1, -1), 0)
            
    return solutions, r6

#def r13(y, e, o, a, r3, fTol, varTols, solutions, r6):
#    n = r3.shape[1] / 2
#    r18, r19 = r3[:, :n], r3[:, n:]
#    r5_L, r5_U =  where(logical_and(r18 < fTol, o[:, :n] == 0.0)), where(logical_and(r19 < fTol, o[:, n:] == 0.0))
#    r4 = 0.5 * (y + e)
#    r20 = 0.5 * (e - y)
#    r5 = []
#    # L
#    for I in range(len(r5_L[0])):#TODO: rework it
#        i, j = r5_L[0][I], r5_L[1][I]
#        tmp = r4[i].copy()
#        tmp[j] -= 0.5*r20[i, j]
#        r5.append(tmp)
#    # U
#    for I in range(len(r5_U[0])):#TODO: rework it
#        i, j = r5_U[0][I], r5_U[1][I]
#        tmp = r4[i].copy()
#        tmp[j] += 0.5*r20[i, j]
#        r5.append(tmp)
#    
#    for c in r5:
#        if len(solutions) == 0 or not any(all(abs(c - r6) < varTols, 1)): 
#            solutions.append(c)
#            #r6 = asarray(solutions)
#            r6 = append(r6, c.reshape(1, -1), 0)
#            
#    return solutions, r6
