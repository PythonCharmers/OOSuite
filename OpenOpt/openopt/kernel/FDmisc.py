# Handling of FuncDesigner probs

from numpy import empty, hstack, vstack, asfarray, all, atleast_1d, cumsum, asarray, zeros,  atleast_2d, ndarray, prod, ones
from nonOptMisc import scipyInstalled, Hstack, Vstack, Find, isspmatrix, SparseMatrixConstructor, DenseMatrixConstructor, Bmat

def setStartVectorAndTranslators(p):
    for fn in ['lb', 'ub', 'A', 'Aeq', 'b', 'beq']:
        if not hasattr(p, fn): continue
        val = getattr(p, fn)
        if val is not None and any(isfinite(val)):
            p.err('while using oovars providing lb, ub, A, Aeq for whole prob is forbidden, use for each oovar instead')
            
    if not isinstance(p.x0, dict):
        p.err('Unexpected start point type: Python dict expected, '+ str(type(p.x0)) + ' obtained')
    startPoint = p.x0
    assert all(asarray([atleast_1d(val).ndim for val in startPoint.values()]) == 1)
    
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
    #point2vector = lambda point: atleast_1d(hstack([asfarray(point[oov]) for oov in optVars]))
    

    def point2vector(point):
        r = []
        for oov in optVars:
            if oov in point:# i.e. in dict keys
                r.append(point[oov])
            else:
                r.append(zeros(asarray(startPoint[oov]).shape))
        return atleast_1d(hstack(r))

    vector_x0 = point2vector(startPoint)
    n = vector_x0.size
    p.n = n
    
    
    oovar_sizes = [asarray(startPoint[elem]).size for elem in optVars]
    oovar_indexes = cumsum([0] + oovar_sizes)
    
    assert len(oovar_indexes) == len(optVars) + 1
    
    #p.oocons = set() # constraints
    
    # TODO: mb use oovarsIndDict here as well (as for derivatives?)
    dictFixed = {}
    if fixedVars is not None:
        for v in fixedVars:
            dictFixed[v] = startPoint[v]
    def vector2point(x):
        r = dictFixed.copy()
        for i, oov in enumerate(optVars):
            r[oov] = x[oovar_indexes[i]:oovar_indexes[i+1]]
        return r

    oovarsIndDict = {}#dictFixed.copy()
    for i, oov in enumerate(optVars):
        #oovarsIndDict[oov.name] = (oovar_indexes[i], oovar_indexes[i+1])
        oovarsIndDict[oov] = (oovar_indexes[i], oovar_indexes[i+1])
        
    def pointDerivative2array(pointDerivarive, asSparse = False,  func=None, point=None): 
        # asSparse can be True, False, 'auto'
        # !!!!!!!!!!! TODO: implement asSparse = 'auto' properly
        if not scipyInstalled and asSparse == 'auto':
            asSparse = False
        if asSparse is not False and not scipyInstalled:
            p.err('to handle sparse matrices you should have module scipy installed') 

        # however, this check is performed in other function (before this one)
        # and those constraints are excluded automaticvally

        if len(pointDerivarive) == 0: 
            if func is not None:
                assert point is not None
                funcLen = func(point).size
                if asSparse:
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
        
        newStyle = 1
        
        if asSparse is not False and newStyle:
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
            if funcLen == 1:
                r = DenseMatrixConstructor(n)
            else:
                if asSparse:
                    r = SparseMatrixConstructor((n, funcLen))
                else:
                    r = DenseMatrixConstructor((n, funcLen))            
            for key, val in pointDerivarive.items():
                # TODO: remove indexes, do as above for sparse 
                indexes = oovarsIndDict[key]
                if not asSparse and isspmatrix(val): val = val.A
                if r.ndim == 1:
                    r[indexes[0]:indexes[1]] = val.flatten()
                else:
                    r[indexes[0]:indexes[1], :] = val.T
            if asSparse and funcLen == 1: 
                return SparseMatrixConstructor(r)
            else: 
                return r.T if r.ndim > 1 else r.reshape(1, -1)
                
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
    p.oovars = optVars # Where it is used?
    p.optVars, p.fixedVars = optVars, fixedVars
    p._point2vector, p._vector2point = point2vector, vector2point
    p._pointDerivative2array = pointDerivative2array
    p._oovarsIndDict = oovarsIndDict
    
    # TODO: replave p.x0 in RunProbSolver finish  
    p._x0, p.x0 = p.x0, vector_x0 
    
    def linearOOFunsToMatrices(oofuns):
        # oofuns should be linear
        C, d = [], []
        Z = p._vector2point(zeros(p.n))
        for elem in oofuns:
            if elem.isConstraint:
                lin_oofun = elem.oofun
            else:
                lin_oofun = elem
            if not lin_oofun.is_linear:
                raise OpenOptException("this function hasn't been intended to work with nonlinear FuncDesigner oofuns")
            C.append(p._pointDerivative2array(lin_oofun._D(Z, **p._D_kwargs), asSparse = 'auto'))
            d.append(-lin_oofun(Z))
            
        if any([isspmatrix(elem) for elem in C]):
            Vstack = scipy.sparse.vstack
        else:
            Vstack = vstack # i.e. numpy.vstack

        C, d = Vstack(C), hstack(d).flatten()

        return C, d    
    p._linearOOFunsToMatrices = linearOOFunsToMatrices
    
