from interalgMisc import *
def r14IP(p, y, e, vv, asdf1, C, CBKPMV, itn, g, nNodes,  \
         frc, fTol, maxSolutions, varTols, solutions, r6, _in, dataType, \
         maxNodes, _s, xRecord):
             
    m, n = y.shape
    
    ip = func10(y, e, vv)
    
    o, a = func8(ip, asdf1, dataType)

    o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T

    if itn == 0: 
        # TODO: fix it
        #_s = atleast_1d(nanmax(a-o))
        _s = atleast_1d(inf)
        
    nodes = func11(y, e, o, a, _s,'IP')

    nodes.sort(key = lambda obj: obj.key)

    if len(_in) == 0:
        an = nodes
    else:
        arr1 = [node.key for node in _in]
        arr2 = [node.key for node in nodes]
        r10 = searchsorted(arr1, arr2)
        an = insert(_in, r10, nodes)


    p.iterfcn(xk=nan, fk=1234, rk = 9876)
    
  
    an, g = func9(an, fo, g, 'IP')
    #nn = 1 if asdf1.isUncycled and all(isfinite(a)) and all(isfinite(o)) and p._isOnlyBoxBounded else maxNodes
    #an, g = func5(an, nn, g)


    y, e, _in, _s = \
                func12(an, self.maxActiveNodes, maxSolutions, solutions, r6, varTols, None, 'IP')# Case=3
                
    return an, g, fo, _s, solutions, r6, xRecord, frc, CBKPMV
