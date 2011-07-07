from interalgLLR import *
from numpy import inf, prod, searchsorted


def r14IP(p, y, e, vv, asdf1, C, CBKPMV, itn, g, nNodes,  \
         frc, fTol, maxSolutions, varTols, solutions, r6, _in, dataType, \
         maxNodes, _s, xRecord):
#    global ITER
#    ITER = p.iter
    required_sigma = p.ftol
    
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
    #nodes.sort(key = lambda obj: obj.volumeResidual, reverse=True)

    if len(_in) == 0:
        an = nodes
    else:
        arr1 = [node.key for node in _in]
        arr2 = [node.key for node in nodes]
#        arr1 = -array([node.volumeResidual for node in _in])
#        arr2 = -array([node.volumeResidual for node in nodes])
        
        r10 = searchsorted(arr1, arr2)
        an = insert(_in, r10, nodes)

    ao_diff = array([node.key for node in an])
    volumes = array([node.volume for node in an])
    r10 = ao_diff <= 0.75*(required_sigma-p._residual) / (prod(p.ub-p.lb) - p._volume)
    #r10 = nanmax(a-o, 1) <= required_sigma / prod(p.ub-p.lb)
    
    ind = where(r10)[0]
    # TODO: use true_sum
    #print sum(array([an[i].F for i in ind]) * array([an[i].volume for i in ind]))
    #print 'p._F:', p._F, 'delta:', sum(array([an[i].F for i in ind]) * array([an[i].volume for i in ind]))
    v = volumes[ind]
    p._F += sum(array([an[i].F for i in ind]) * v)
    residuals = ao_diff[ind] * v
    p._residual += residuals.sum()
    p._volume += v.sum()
    
    #print 'iter:', p.iter, 'nNodes:', len(an), 'F:', p._F, 'div:', ao_diff / (required_sigma / prod(p.ub-p.lb))
    an = array(an, object)
    an = take(an, where(logical_not(r10))[0])
    nNodes.append(len(an))
   
    p.iterfcn(xk=array(nan), fk=p._F, rk = 0)#TODO: change rk to something like p._r0 - p._residual
    if p.istop != 0: 
        ao_diff = array([node.key for node in an])
        volumes = array([node.volume for node in an])
        p._residual += sum(ao_diff * volumes)
        _s = None
 
    #an, g = func9(an, fo, g, 'IP')
    #nn = 1 if asdf1.isUncycled and all(isfinite(a)) and all(isfinite(o)) and p._isOnlyBoxBounded else maxNodes
    #an, g = func5(an, nn, g)

    return an, g, inf, _s, solutions, r6, xRecord, frc, CBKPMV
