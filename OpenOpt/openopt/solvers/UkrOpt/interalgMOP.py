from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, \
nan, isinf, arange, vstack, inf, where, logical_not, take, argmax, argmin, min, abs, hstack, empty, insert, \
isfinite, append, atleast_2d, prod, logical_xor, argsort, asfarray, ones, log2, zeros, log1p

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

def r14MOP(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, itn, g, nNodes,  \
         r41, fTol, Solutions, varTols, _in, dataType, \
         maxNodes, _s, indTC, xRecord):

    assert p.probType == 'MOP'
    
    if len(p._discreteVarsNumList):
        adjustDiscreteVarBounds(y, e, p)
    
    if itn == 0: 
        # TODO: change for constrained probs
        _s = atleast_1d(inf)
    
    ol, al = [], []
    targets = p.targets # TODO: check it
    m, n = y.shape
    ol, al = [[] for k in range(m)], [[] for k in range(m)]
    for i, t in enumerate(targets):
        o, a, definiteRange = func82(y, e, vv, t.func, dataType)
        o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T
        for j in range(m):
            ol[j].append(o[j])
            al[j].append(a[j])
        #ol.append(o.reshape(2*n, m).T.tolist())
        #al.append(a.reshape(2*n, m).T.tolist())

    nlhf = r43(targets, Solutions, ol, al, vv, dataType)
    
    # DEBUG!
#    from numpy import array_equal, isinf
#    nlhf2 = r43(targets, Solutions, ol, al, vv, dataType, Case=2)
#    if nlhf is not None:
#        nlhf3, nlhf4 = nlhf.copy(), nlhf2.copy()
#        nlhf3[isinf(nlhf3)] = 0.001
#        nlhf4[isinf(nlhf4)] = 0.001
#        if not array_equal(nlhf3, nlhf4):
#            pass
    # Debug end
    
    fo_prev = 0
    # TODO: remove NaN nodes here

    if y.size == 0:
        return _in, g, fo_prev, _s, Solutions, xRecord, r41, r40
    
    nodes = func11(y, e, nlhc, indTC, residual, ol, al, _s, p)
    
    #y, e = func4(y, e, o, a, fo)
    
    newNLH = False
    
    assert p.solver.dataHandling == 'raw', '"sorted" mode is unimplemented for MOP yet'
    tnlh_curr = nlhf
    if nlhc is not None: 
        tnlh_curr += nlhc

    asdf1 = [t.func for t in p.targets]
    r5F, r5Coords = getr4Values(vv, y, e, tnlh_curr, asdf1, C, p.contol, dataType, p) 

    # debug!!
#    if itn > 50:
#        r5Coords += [[1, 2, 3, 4]]
#        r5F += [[2.96, -1.36]]
    # debug end
    
    r44(Solutions, r5Coords, r5F, targets)
    fo = 0 # unused for MOP
    
    # TODO: better of nlhc for unconstrained probs
    if len(_in) != 0:
        ol2 = [node.o for node in _in]
        al2 = [node.a for node in _in]
        nlhc2 = [node.nlhc for node in _in]
#        o2, a2 = [node.o for node in _in], [node.a for node in _in]
        tnlh2 = r43(targets, Solutions, ol2, al2, vv, dataType)
        if nlhc2[0] is not None:
            tnlh2 += asarray(nlhc2)
        tnlh_all = vstack((tnlh_curr, tnlh2))

    else:
        tnlh_all = tnlh_curr
    
    T1, T2 = tnlh_all[:, :tnlh_all.shape[1]/2], tnlh_all[:, tnlh_all.shape[1]/2:]
    T = where(logical_or(T1 < T2, isnan(T2)), T1, T2)
    t = nanargmin(T, 1)
    p._t = t
    w = arange(t.size)
    n = p.n
    p.__s = \
    nanmin(vstack(([tnlh_all[w, t], tnlh_all[w, n+t]])), 0)

    NN = T[w, t].flatten()
    #NN = nanmin(tnlh_all, 1)
    r10 = logical_or(isnan(NN), NN == inf)
    
    an = hstack((nodes,  _in))
    #print NN
    
    if any(r10):
        ind = where(logical_not(r10))[0]
        
        # Debug
#        ind2 = where(r10)[0]
#        for i in ind2:
#            if all (an[i].y <= 0 ) and all(an[i].e >= 0 ):
##                raise 0
#                pass
        # debug end
        
        an = take(an, ind, axis=0, out=an[:ind.size])
        #tnlh = take(tnlh, ind, axis=0, out=tnlh[:ind.size])
        NN = take(NN, ind, axis=0, out=NN[:ind.size])
    
    astnlh = argsort(NN)
    an = an[astnlh]
    #print len(an)
    
#    else: #if p.solver.dataHandling == 'sorted':
#        p.err('dataHandling = "sorted" is unimplemented yet for MOP')

#        p._nObtainedSolutions = len(solutions)
#        if p._nObtainedSolutions > maxSolutions:
#            solutions = solutions[:maxSolutions]
#            p.istop = 0
#            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
#            return an, g, fo, None, solutions, coords, xRecord, r41, r40
    
    #p.iterfcn(xk, Min)
    #p.iterfcn(xRecord, r40)
    
    # TODO: fix it
    #p.iterfcn(p._x0)
    print('iter: %d (%d) frontLenght: %d' %(p.iter, itn, len(Solutions.coords)))
    
    if p.istop != 0: 
        return an, g, fo, None, Solutions, xRecord, r41, r40
        
    #an, g = func9(an, fo, g, p)

    nn = maxNodes#1 if asdf1.isUncycled and all(isfinite(o)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes
    
    an, g = func5(an, nn, g, p)
    nNodes.append(len(an))
    return an, g, fo, _s, Solutions, xRecord, r41, r40



def r44(Solutions, r5Coords, r5F, targets):
#    print Solutions.F
#    if len(Solutions.F) != Solutions.coords.shape[0]:
#        raise 0
    # TODO: rework it
    #sf = asarray(Solutions.F)
    
    m= len(r5Coords)
    n = len(r5Coords[0])
    # TODO: mb use inplace r5Coords / r5F modification instead?
#    S = m+1
    for j in range(m):
#        S -= 1
        if Solutions.coords.size == 0:
            Solutions.coords = array(r5Coords[j])
            Solutions.F.append(r5F[0])
            continue
        M = Solutions.coords.shape[0] 
        
        r47 = empty(M, bool)
        r47.fill(False)
        r48 = empty(M, bool)
        r48.fill(False)
        for i, target in enumerate(targets):
            
            f = r5F[j][i]
            
            # TODO: rewrite it
            d = f - asarray([Solutions.F[k][i] for k in range(M)]) # vector-matrix
            
            val, tol = target.val, target.tol
            if val == inf:
                r52 = d > tol
                r36olution_better = d < 0#-tol
            elif val == -inf:
                r52 = d < -tol
                r36olution_better = d > 0#tol
            else:
                r20 = abs(f - target) - abs(Solutions.F[i] - target)
                r52 = r20 < tol # abs(f[i] - target)  < abs(Solutions.F[i] - target) + tol
                r36olution_better = r20 > 0#-tol # abs(Solutions.F[i] - target)  < abs(f[i] - target) + tol

            r47 = logical_or(r47, r52)
            r48 = logical_or(r48, r36olution_better)
        
        accept_c = all(r47)
        #print sum(asarray(Solutions.F))/asarray(Solutions.F).size
        if accept_c:
            r49 = logical_not(r48)
            remove_s = any(r49)
            if remove_s:# and False :
                r50 = where(r49)[0]
                Solutions.coords[r50[0]] = r5Coords[j]
                Solutions.F[r50[0]] = r5F[j]
                
                if r50.size > 1:
                    r49[r50[0]] = False
                    indLeft = logical_not(r49)
                    indLeftPositions = where(indLeft)[0]
                    newSolNumber = Solutions.coords.shape[0] - r50.size + 1
                    Solutions.coords = take(Solutions.coords, indLeftPositions, axis=0, out = Solutions.coords[:newSolNumber])
                    solutionsF2 = asarray(Solutions.F, object)
                    solutionsF2 = take(solutionsF2, indLeftPositions, axis=0, out = solutionsF2[:newSolNumber])
                    Solutions.F = solutionsF2.tolist()
            else:
                Solutions.coords = vstack((Solutions.coords, r5Coords[j]))
                Solutions.F.append(r5F[j])
#    r0 = 1000
#    for i in range(len(Solutions.F)):
#        for j in range(i):
#            r = max(array(Solutions.F[i])-array(Solutions.F[j]))
#            r0 = min((r, r0))
#    print 'r0:', r0

#        else:# Debug
#            pass
    

def r43(targets, Solutions, lf, uf, vv, dataType):
    lf, uf = asarray(lf), asarray(uf)
    solutionsF = Solutions.F
    S = len(solutionsF)
    if S == 0: return None
    
    #print '!', asarray(lf).shape, asarray(uf).shape
    #lf, uf = asarray(lf), asarray(uf)
    
    m = len(lf)
    n = lf[0][0].size/2
    r = zeros((m, 2*n))

    for _s in solutionsF:
        s = asarray(_s)
        tmp = ones((m, 2*n))
        for i, t in enumerate(targets):
            f, val, tol = t.func, t.val, t.tol
            
            #TODO: mb optimize it
            o, a = lf[:, i], uf[:, i] 
            
            if val == inf:
                ff = s[i] + tol
                ind = a > ff
                if any(ind):
                    t1 = a[ind]
                    t2 = o[ind].copy()
                    t2[t2>ff] = ff
                    
                    # TODO: check discrete cases
                    tmp[ind] *= (ff-t2) / (t1-t2)
                    
            elif val == -inf:
                ff = s[i] - tol
                ind = o < ff
                if any(ind):
                    t1 = a[ind].copy()
                    t2 = o[ind]
                    t1[t1<ff] = ff
                    # TODO: check discrete cases
                    tmp[ind] *= (t1-ff) / (t1-t2)
            else:
                raise('unimplemented yet')    
#            if any(tmp<0) or any(tmp>1):
#                raise 0

        #r -= log2(1.0 - tmp)
        r -= log1p(-tmp) * 1.4426950408889634 # log2(e)
                
    return r

#def r432(targets, Solutions, lf, uf, vv, dataType):
#    solutionsF = Solutions.F
#    S = len(solutionsF)
#    if S == 0: return None
#    
#    #print '!', asarray(lf).shape, asarray(uf).shape
#    #lf, uf = asarray(lf), asarray(uf)
#    
#    m = len(lf)
#    n = lf[0][0].size/2
#    r = zeros((m, 2*n))
#
#    for _s in solutionsF:
#        s = asarray(_s)
#        for j in range(m):
#            o_, a_ = lf[j], uf[j]
#            tmp = ones(2*n)
#            for i, t in enumerate(targets):
#                f, val, tol = t.func, t.val, t.tol
#                #F = asarray([solutionsF[k][i] for k in range(K)])
#                o, a = o_[i], a_[i]
#                
#                if val == inf:
#                    ff = s[i] + tol
#                    ind = a > ff
#                    if any(ind):
#                        t1 = a[ind]
#                        t2 = o[ind].copy()
#                        t2[t2>ff] = ff
#                        
#                        # TODO: check discrete cases
#                        tmp[ind] *= (ff-t2) / (t1-t2)
#                        
#                elif val == -inf:
#                    ff = s[i] - tol
#                    ind = o < ff
#                    if any(ind):
#                        t1 = a[ind].copy()
#                        t2 = o[ind]
#                        t1[t1<ff] = ff
#                        
#                        # TODO: check discrete cases
#                        tmp[ind] *= (t1-ff) / (t1-t2)
#                else:
#                    raise('unimplemented yet')    
##            if any(tmp<0) or any(tmp>1):
##                raise 0
#            tmp[tmp<0.0] = 0.0 # suppress roundoff errors
#            r[j] -= log2(1.0 - tmp)
#                
#    return r
    
