# Handling of FuncDesigner probs
from numpy import empty, hstack, vstack, asfarray, all, atleast_1d, cumsum, asarray, zeros,  atleast_2d, ndarray, prod, ones, copy, nan, flatnonzero, array_equal
from nonOptMisc import scipyInstalled, Hstack, Vstack, Find, isspmatrix, SparseMatrixConstructor, DenseMatrixConstructor, Bmat

def setStartVectorAndTranslators(p):
    startPoint = p.x0
    #assert all(asarray([atleast_1d(val).ndim for val in startPoint.values()]) == 1)
    
    # !!!! TODO: handle fixed oovars
    #oovars = list(startPoint.keys())
    
    fixedVars, optVars = None, None
    
    if p.optVars is not None:
        if type(p.optVars) not in [list, tuple]:
            assert hasattr(p.optVars, 'is_oovar')
            p.optVars = [p.optVars]
            optVars = p.optVars
        else:
            optVars = list(p.optVars)
        fixedVars = list(set(startPoint.keys()).difference(set(optVars)))
        p.fixedVars = fixedVars
    elif p.fixedVars is not None:
        if type(p.fixedVars) not in [list, tuple]:
            assert hasattr(p.fixedVars, 'is_oovar')
            p.fixedVars = [p.fixedVars]
            fixedVars = p.fixedVars
        else:
            fixedVars = list(p.fixedVars)
        optVars = list(set(startPoint.keys()).difference(set(fixedVars)))
        p.optVars = optVars
    else:
        optVars = startPoint.keys()
    
    p._fixedVars = set(fixedVars) if fixedVars is not None else set()
    p._optVars = set(optVars) if optVars is not None else set()
        
    # point should be FuncDesigner point that currently is Python dict        
    # point2vector = lambda point: atleast_1d(hstack([asfarray(point[oov]) for oov in optVars]))
    
    p._optVarSizes = dict([(oov, asarray(startPoint[oov]).size) for oov in optVars])
    point2vector = lambda point: atleast_1d(hstack([(point[oov] if oov in point else zeros(sizes[oov])) for oov in p._optVarSizes]))
    # 2nd case can trigger from objective/constraints defined over some of opt oovars only
        
    vector_x0 = point2vector(startPoint)
    n = vector_x0.size
    p.n = n
    
    oovar_sizes = [asarray(startPoint[elem]).size for elem in optVars]
    oovar_indexes = cumsum([0] + oovar_sizes)
    
    # TODO: mb use oovarsIndDict here as well (as for derivatives?)
    from FuncDesigner import oopoint
    startDictData = [] if fixedVars is None else [(v, startPoint[v]) for v in fixedVars]

    #vector2point = lambda x: oopoint(startDictData + [(oov, x[oovar_indexes[i]:oovar_indexes[i+1]]) for i, oov in enumerate(optVars)])
    p._FDtranslator = {'prevX':nan}
    def vector2point(x): 
        x = atleast_1d(asfarray(x)).copy()
        if array_equal(x, p._FDtranslator['prevX']):
            return p._FDtranslator['prevVal']
            
        # without copy() ipopt and probably others can replace it by noise after closing
        r = oopoint(startDictData + \
                    [(oov, x[oovar_indexes[i]:oovar_indexes[i+1]]) for i, oov in enumerate(optVars)])
        
        p._FDtranslator['prevVal'] = r 
        p._FDtranslator['prevX'] = copy(x)

        return r

    oovarsIndDict = dict([(oov, (oovar_indexes[i], oovar_indexes[i+1])) for i, oov in enumerate(optVars)])
        
    def pointDerivative2array(pointDerivarive, useSparse = 'auto',  func=None, point=None): 
        
        # useSparse can be True, False, 'auto'
        if not scipyInstalled and useSparse == 'auto':
            useSparse = False
        if useSparse is True and not scipyInstalled:
            p.err('to handle sparse matrices you should have module scipy installed') 

        if len(pointDerivarive) == 0: 
            if func is not None:
                assert point is not None
                funcLen = func(point).size
                if useSparse is not False:
                    return SparseMatrixConstructor((funcLen, n))
                else:
                    return DenseMatrixConstructor((funcLen, n))
            else:
                p.err('unclear error, maybe you have constraint independend on any optimization variables') 

        key, val = pointDerivarive.items()[0]
        
        if isinstance(val, float) or (isinstance(val, ndarray) and val.shape == ()):
            val = atleast_1d(val)
        var_inds = oovarsIndDict[key]
        # val.size works in other way (as nnz) for scipy.sparse matrices
        funcLen = int(round(prod(val.shape) / (var_inds[1] - var_inds[0]))) 
        
        # CHANGES
        
        # 1. Calculate number of zero/nonzero elements
        involveSparse = useSparse
        if useSparse == 'auto':
            nTotal = n * funcLen#sum([prod(elem.shape) for elem in pointDerivarive.values()])
            nNonZero = sum([(elem.size if isspmatrix(elem) else len(flatnonzero(asarray(elem)))) for elem in pointDerivarive.values()])
            involveSparse = 4*nNonZero < nTotal and nTotal > 1000
        # 2. Create init result matrix
#        if funcLen == 1:
#            r = DenseMatrixConstructor(n)
#        # TODO: uncomment and implement!
##        elif not involveSparse:
##            r = DenseMatrixConstructor((n, funcLen))
#        else:
#            I, J, Vals = [], [], []
#            for key, val in pointDerivarive.items():
#                if isspmatrix(val):
#                    _i, _j, _vals = Find(val)
#                    I += _i
#                    J += _j
#                    Vals += _vals
#                else:
#                    _i, _j = nonzero(val)
#                    
#                    I += range(atleast_2d(val).shape[0]) * funcLen
#                    J += range(funcLen) * (atleast_2d(val).shape[1])
#                    Vals += val.flatten().tolist()
#            r = SparseMatrixConstructor((Vals,  (I, J)), shape = (n, funcLen))
#            return r
#            
                #r[indexes[0]:indexes[1], :] = val.T
            #r = SparseMatrixConstructor((n, funcLen)) 
        
#        if funcLen == 1:
#            r = DenseMatrixConstructor(n)
#        else:
#            if useSparse:
#                r = SparseMatrixConstructor((n, funcLen))
#            else:
#                r = DenseMatrixConstructor((n, funcLen))            
        # CHANGES END
        
        if involveSparse:# and newStyle:
            # USE STACK
            r2 = []
            hasSparse = False
            for i, var in enumerate(optVars):
                if var in pointDerivarive:#i.e. one of its keys
                    tmp = pointDerivarive[var]
                    if isspmatrix(tmp): hasSparse = True
                    if isinstance(tmp, float) or (isinstance(tmp, ndarray) and tmp.shape == ()):
                        tmp = atleast_1d(tmp)
                    if tmp.ndim < 2:
                        tmp = tmp.reshape(funcLen, prod(tmp.shape) // funcLen)
                    r2.append(tmp)
                else:
                    r2.append(SparseMatrixConstructor((funcLen, oovar_sizes[i])))
                    hasSparse = True
            r3 = Hstack(r2) if hasSparse else hstack(r2)
            if isspmatrix(r3) and r3.nnz > 0.25 * prod(r3.shape): r3 = r3.A
            return r3
        else:
            # USE INSERT
            if funcLen == 1:
                r = DenseMatrixConstructor(n)
            else:
                r = SparseMatrixConstructor((funcLen, n))if involveSparse else DenseMatrixConstructor((funcLen, n)) 
            for key, val in pointDerivarive.items():
                # TODO: remove indexes, do as above for sparse 
                indexes = oovarsIndDict[key]
                if not involveSparse and isspmatrix(val): val = val.A
                if r.ndim == 1:
                    r[indexes[0]:indexes[1]] = val.flatten() if type(val) == ndarray else val
                else:
                    r[:, indexes[0]:indexes[1]] = val.reshape(funcLen, prod(val.shape)/funcLen)
            if useSparse is True and funcLen == 1: 
                return SparseMatrixConstructor(r)
            elif r.ndim <= 1:
                r = r.reshape(1, -1)
            if useSparse is False and hasattr(r, 'toarray'):
                r = r.toarray()
            return r
                
                
    def getPattern(oofuns):
        # oofuns is Python list of oofuns
        # result is 1d-array, so we can omit using sparsity here and involve it in an upper stack level func
        assert isinstance(oofuns, list), 'oofuns should be Python list, inform developers of the bug'
        R = []
        for oof in oofuns:
            r = []
            dep = oof._getDep()
            for oov in optVars:
                constructor = ones if oov in dep else SparseMatrixConstructor
                r.append(constructor((1, asarray(startPoint[oov]).size)))
            if any([isspmatrix(elem) for elem in r]):
                rr = Hstack(r) if len(r) > 1 else r[0]
            elif len(r)>1:
                rr = hstack(r)
            else:
                rr = r[0]
            SIZE = asarray(oof(startPoint)).size
            if SIZE > 1:  rr = Vstack([rr]*SIZE)  if isspmatrix(rr) else vstack([rr]*SIZE)
            R.append(rr)
        result = Vstack(R) if any([isspmatrix(_r) for _r in R]) else vstack(R)
        
        return result
        
    p._getPattern = getPattern
    p.optVars, p.fixedVars = optVars, fixedVars
    p._point2vector, p._vector2point = point2vector, vector2point
    p._pointDerivative2array = pointDerivative2array
    p._oovarsIndDict = oovarsIndDict
    
    # TODO: replave p.x0 in RunProbSolver finish  
    p._x0, p.x0 = p.x0, vector_x0 
    
    def linearOOFunsToMatrices(oofuns): #, useSparse = 'auto'
        # oofuns should be linear
        C, d = [], []
        Z = p._vector2point(zeros(p.n))
        for elem in oofuns:
            if elem.isConstraint:
                lin_oofun = elem.oofun
            else:
                lin_oofun = elem
            if lin_oofun.getOrder(self.optVars, self.fixedVars) > 1:
                raise OpenOptException("this function hasn't been intended to work with nonlinear FuncDesigner oofuns")
            C.append(p._pointDerivative2array(lin_oofun.D(Z, **p._D_kwargs), useSparse = p.useSparse))
            d.append(-lin_oofun(Z))
            
        if any([isspmatrix(elem) for elem in C]):
            Vstack = scipy.sparse.vstack
        else:
            Vstack = vstack # i.e. numpy.vstack

        C, d = Vstack(C), hstack(d).flatten()

        return C, d    
    p._linearOOFunsToMatrices = linearOOFunsToMatrices
    
