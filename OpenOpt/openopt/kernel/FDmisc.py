# Handling of FuncDesigner probs

from numpy import empty, hstack, asfarray, all, atleast_1d, cumsum, asarray, zeros,  atleast_2d

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
        pkeys = set(point.keys()) # elseware it's too slow
        for oov in optVars:
            if oov in pkeys:
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
        oovarsIndDict[oov.name] = (oovar_indexes[i], oovar_indexes[i+1])

    def pointDerivative2array(pointDerivarive): 
        if len(pointDerivarive) == 0: 
            p.err('unclear error, mb you have constraint independend on any optimization variables') 
            
        name, val = pointDerivarive.items()[0]
        var_inds = oovarsIndDict[name]
        funcLen = val.size / (var_inds[1] - var_inds[0])
        if funcLen == 1:
            r = zeros(n)
        else:
            d1 = funcLen
            r = zeros((n, d1))
        #from numpy import any, diff
        #assert not any(diff([(pointDerivarive.values()[j]).shape[1] for j in xrange(len(pointDerivarive))]))
        
        for key, val in pointDerivarive.items():
            indexes = oovarsIndDict[key]
            r[indexes[0]:indexes[1]] = val
        return r.T if r.ndim > 1 else r.reshape(1, -1)
        
        
    
    p.oovars = optVars # Where it is used?
    p.optVars, p.fixedVars = optVars, fixedVars
    p._point2vector, p._vector2point = point2vector, vector2point
    p._pointDerivative2array = pointDerivative2array
    p._oovarsIndDict = oovarsIndDict
    
    # TODO: replave p.x0 in RunProbSolver finish  
    p._x0, p.x0 = p.x0, vector_x0 
    
    
    
    #############################################
#    p.oovars = set()
#    p.oofuns = set()
#    for FuncType in ['f', 'c', 'h']:
#        Funcs = getattr(p, FuncType)
#        if Funcs is None: continue
#        if isinstance(Funcs, oofun):
#            Funcs._recursivePrepare(p)
#        else:
#            if type(Funcs) not in [tuple, list]:
#                p.err('when x0 is absent, oofuns (with oovars) are expected')
#            for fun in Funcs:
#                if not isinstance(fun, oofun):
#                    p.err('when x0 is absent, oofuns (with oovars) are expected')
#                fun._recursivePrepare(p)
#    assert len(p.oovars) > 0
#    n = 0
#    for fn in ['x0', 'lb', 'ub', 'A', 'Aeq', 'b', 'beq']:
#        if not hasattr(p, fn): continue
#        val = getattr(p, fn)
#        if val is not None and any(isfinite(val)):
#            p.err('while using oovars providing x0, lb, ub, A, Aeq for whole prob is forbidden, use for each oovar instead')
#
#    x0, lb, ub = [], [], []
#
#    for var in p.oovars:
#        var.dep = range(n, n+var.size)
#        n += var.size
#        x0 += list(atleast_1d(asarray(var.v0)))
#        lb += list(atleast_1d(asarray(var.lb)))
#        ub += list(atleast_1d(asarray(var.ub)))
#    p.n = n
#    p.x0 = x0
#    p.lb = lb
#    p.ub = ub
