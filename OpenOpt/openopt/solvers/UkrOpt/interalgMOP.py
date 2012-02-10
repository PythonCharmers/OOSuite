PythonSum = sum
from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, \
nan, isinf, arange, vstack, inf, where, logical_not, take, argmax, argmin, min, abs, hstack, empty, insert, \
isfinite, append, atleast_2d, prod, logical_xor, argsort, asfarray, ones, log2, zeros, log1p, array_split

from interalgLLR import *

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax


def r43_seq(Arg):
    targets_vals, targets_tols, solutionsF, lf, uf = Arg
    lf, uf = asarray(lf), asarray(uf)

    S = len(solutionsF)
    if S == 0: return None

    m = len(lf)
    n = lf.shape[2]/2
    r = zeros((m, 2*n))

    for _s in solutionsF:
        s = atleast_1d(_s)
        tmp = ones((m, 2*n))
        for i in range(len(targets_vals)):
            val, tol = targets_vals[i], targets_tols[i]#, t.val, t.tol
            #TODO: mb optimize it
            o, a = lf[:, i], uf[:, i] 
            if val == inf:
                ff = s[i] + tol
                ind = a > ff
                if any(ind):
                    t1 = a[ind]
                    t2 = o[ind]
                    
                    # TODO: check discrete cases
                    Tmp = (ff-t2) / (t1-t2)
                    Tmp[t1==t2] = 0.0 # for discrete cases
                    tmp[ind] *= Tmp
                    tmp[ff<o] = 0.0
                    
            elif val == -inf:
                ff = s[i] - tol
                ind = o < ff
                if any(ind):
                    t1 = a[ind]
                    t2 = o[ind]
                    # TODO: check discrete cases
                    Tmp = (t1-ff) / (t1-t2)
                    Tmp[t1==t2] = 0.0 # for discrete cases
                    tmp[ind] *= Tmp
                    tmp[a<ff] = 0.0
            else: # finite val
                ff = abs(s[i]-val) - tol
                if ff <= 0:
                    continue
                _lf, _uf = o - val, a - val
                ind = logical_or(_lf < ff, _uf > - ff)
                _lf = _lf[ind]
                _uf = _uf[ind]
                _lf[_lf>ff] = ff
                _lf[_lf<-ff] = -ff
                _uf[_uf<-ff] = -ff
                _uf[_uf>ff] = ff
                
                r20 = a[ind] - o[ind]
                Tmp = 1.0 - (_uf - _lf) / r20
                Tmp[r20==0] = 0.0 # for discrete cases
                tmp[ind] *= Tmp
                #raise('unimplemented yet')    
#            if any(tmp<0) or any(tmp>1):
#                raise 0
        r -= log1p(-tmp) * 1.4426950408889634 # log2(e)
    return r

from multiprocessing import Pool
def r43(targets, SolutionsF, lf, uf, pool, nProc):
    lf, uf = asarray(lf), asarray(uf)
    target_vals = [t.val for t in targets]
    target_tols = [t.tol for t in targets]
    if pool is None:
        return r43_seq((target_vals, target_tols, SolutionsF, lf, uf))
    #Args = [(target_vals, target_tols, [s], lf, uf) for s in SolutionsF]
    ss = array_split(SolutionsF, nProc)
    #print ss
    Args = [(target_vals, target_tols, s, lf, uf) for s in ss]
    
    result = pool.imap_unordered(r43_seq, Args)#, callback = cb)    
    #result = pool.map(r43_seq, Args)
    r = [elem for elem in result if elem is not None]
    return PythonSum(r)


def r14MOP(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, itn, g, nNodes,  \
         r41, fTol, Solutions, varTols, _in, dataType, \
         maxNodes, _s, indTC, xRecord):

    assert p.probType == 'MOP'
    
    if len(p._discreteVarsNumList):
        adjustDiscreteVarBounds(y, e, p)
    
    if itn == 0: 
        # TODO: change for constrained probs
        _s = atleast_1d(inf)
        if p.nProc != 1:
            p.pool = Pool(processes = p.nProc)
        else:
            p.pool = None
    
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

    nlhf = r43(targets, Solutions.F, ol, al, p.pool, p.nProc)
    
    # DEBUG!
#    from numpy import array_equal, isinf
#    nlhf2 = r43(targets, Solutions.F, ol, al,Case=2)
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
    
    if nlhf is None:
        tnlh_curr = nlhc
    elif nlhc is None: 
        tnlh_curr = nlhf
    else:
        tnlh_curr = nlhf + nlhc

    asdf1 = [t.func for t in p.targets]
    r5F, r5Coords = getr4Values(vv, y, e, tnlh_curr, asdf1, C, p.contol, dataType, p) 

    # debug!!
#    if itn > 50:
#        r5Coords += [[1, 2, 3, 4]]
#        r5F += [[2.96, -1.36]]
    # debug end
    
    nIncome, nOutcome = r44(Solutions, r5Coords, r5F, targets)
    fo = 0 # unused for MOP
    
    # TODO: better of nlhc for unconstrained probs
    
    
    
    if len(_in) != 0:
        an = hstack((nodes,  _in))
    else:
        an = atleast_1d(nodes)
        
    ol2 = [node.o for node in an]
    al2 = [node.a for node in an]
    nlhc2 = [node.nlhc for node in an]
    nlhf2 = r43(targets, Solutions.F, ol2, al2, p.pool, p.nProc)
    tnlh_all = asarray(nlhc2) if nlhf2 is None else nlhf2 if nlhc2[0] is None else asarray(nlhc2) + nlhf2
    
    T1, T2 = tnlh_all[:, :tnlh_all.shape[1]/2], tnlh_all[:, tnlh_all.shape[1]/2:]
    T = where(logical_or(T1 < T2, isnan(T2)), T1, T2)
    t = nanargmin(T, 1)
    p._t = t
    
    nlhf_fixed = asarray([node.nlhf for node in an])
    tnlh_fixed = asarray(nlhc2) if nlhf_fixed is None else nlhf_fixed if nlhc2[0] is None else asarray(nlhc2) + nlhf_fixed
    w = arange(t.size)
    n = p.n
    T = tnlh_all
    p.__s = \
    nanmin(vstack(([T[w, t], T[w, n+t]])), 0)

    NN = T[w, t].flatten()
    #NN = nanmin(tnlh_all, 1)
    r10 = logical_or(isnan(NN), NN == inf)
    
    
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
    p._frontLength = len(Solutions.F)
    p._nIncome = nIncome
    p._nOutcome = nOutcome
    p.iterfcn(p.x0)
    #print('iter: %d (%d) frontLenght: %d' %(p.iter, itn, len(Solutions.coords)))
    
    if p.istop != 0: 
        return an, g, fo, None, Solutions, xRecord, r41, r40
        
    #an, g = func9(an, fo, g, p)

    nn = maxNodes#1 if asdf1.isUncycled and all(isfinite(o)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes
    
    an, g = func5(an, nn, g, p)
    nNodes.append(len(an))
    return an, g, fo, _s, Solutions, xRecord, r41, r40



def r44(Solutions, r5Coords, r5F, targets, sigma = 0.1):
#    print Solutions.F
#    if len(Solutions.F) != Solutions.coords.shape[0]:
#        raise 0
    # TODO: rework it
    #sf = asarray(Solutions.F)
    nIncome, nOutcome = 0, 0
    m= len(r5Coords)
    n = len(r5Coords[0])
    # TODO: mb use inplace r5Coords / r5F modification instead?
    for j in range(m):
        if isnan(r5F[0][0]):
            continue
        if Solutions.coords.size == 0:
            Solutions.coords = array(r5Coords[j]).reshape(1, -1)
            Solutions.F.append(r5F[0])
            nIncome += 1
            continue
        M = Solutions.coords.shape[0] 
        
        r47 = empty(M, bool)
        r47.fill(False)
#        r48 = empty(M, bool)
#        r48.fill(False)
        for i, target in enumerate(targets):
            
            f = r5F[j][i]
            
            # TODO: rewrite it
            F = asarray([Solutions.F[k][i] for k in range(M)])
            #d = f - F # vector-matrix
            
            val, tol = target.val, target.tol
            Tol = sigma * tol
            if val == inf:
                r52 = f > F + Tol
#                r36olution_better = f <= F#-tol
            elif val == -inf:
                r52 = f < F - Tol
#                r36olution_better = f >= F#tol
            else:
                r52 = abs(f - val) < abs(F - val) - Tol
#                r36olution_better = abs(f - val) >= abs(Solutions.F[i] - val)#-tol # abs(Solutions.F[i] - target)  < abs(f[i] - target) + tol
            
            r47 = logical_or(r47, r52)
#            r48 = logical_or(r48, r36olution_better)
        
        accept_c = all(r47)
        #print sum(asarray(Solutions.F))/asarray(Solutions.F).size
        if accept_c:
            nIncome += 1
            #new
            r48 = empty(M, bool)
            r48.fill(False)
            for i, target in enumerate(targets):
                f = r5F[j][i]
                F = asarray([Solutions.F[k][i] for k in range(M)])
                val, tol = target.val, target.tol
                if val == inf:
                    r36olution_better = f < F
                elif val == -inf:
                    r36olution_better = f > F
                else:
                    r36olution_better = abs(f - val) > abs(F - val)
                    #assert 0, 'unimplemented yet'
                r48 = logical_or(r48, r36olution_better)

            r49 = logical_not(r48)
            remove_s = any(r49)
            if remove_s:# and False :
                r50 = where(r49)[0]
                nOutcome += r50.size
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
    return nIncome, nOutcome



#r43 = lambda targets, SolutionsF, lf, uf: r43_seq([t.val for t in targets], [t.tol for t in targets], SolutionsF, lf, uf)
    #parallel_r43(targets, SolutionsF, lf, uf, 2)
    




        
        
#def r43(targets, Solutions, lf, uf):
#    lf, uf = asarray(lf), asarray(uf)
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
#    ss = asarray(solutionsF)
#    tmp = ones((S, m, 2*n))
#    for i, t in enumerate(targets):
#        val, tol = t.val, t.tol
#        
#        #TODO: mb optimize it
#        o, a = tile(lf[:, i], (S, 1, 1)), tile(uf[:, i], (S, 1, 1)) 
#        
#        if val == inf:
#            ff = s[i] + tol
#            ind = a > ff
#            if any(ind):
#                t1 = a[ind]
#                t2 = o[ind]
#                
#                # TODO: check discrete cases
#                Tmp = (ff-t2) / (t1-t2)
#                Tmp[t1==t2] = 0.0 # for discrete cases
#                tmp[ind] *= Tmp
#                tmp[ff<o] = 0.0
#                
#        elif val == -inf:
#            ff = (ss[:, i] - tol).reshape(S, 1, 1)
#            try:
#                ind = o < ff
#            except:
#                pass
#            if any(ind):
#                try:
#                    t1 = a[ind]
#                except:
#                    pass
#                t2 = o[ind]
#                # TODO: check discrete cases
#                Tmp = (t1-ff) / (t1-t2)
#                Tmp[t1==t2] = 0.0 # for discrete cases
#                try:
#                    tmp[ind] *= Tmp
#                except:
#                    pass
#                tmp[a<ff] = 0.0
#        else: # finite val
#            ff = abs(s[i]-val) - tol
#            if ff <= 0:
#                continue
#            _lf, _uf = o - val, a - val
#            ind = logical_or(_lf < ff, _uf > - ff)
#            _lf = _lf[ind]
#            _uf = _uf[ind]
#            _lf[_lf>ff] = ff
#            _lf[_lf<-ff] = -ff
#            _uf[_uf<-ff] = -ff
#            _uf[_uf>ff] = ff
#            
#            r20 = a[ind] - o[ind]
#            Tmp = 1.0 - (_uf - _lf) / r20
#            Tmp[r20==0] = 0.0 # for discrete cases
#            tmp[ind] *= Tmp
#                #raise('unimplemented yet')    
##            if any(tmp<0) or any(tmp>1):
##                raise 0
#        r -= log1p(-sum(tmp, axis=0)) * 1.4426950408889634 # log2(e)
#    return r
