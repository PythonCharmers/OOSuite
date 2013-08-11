from interalgLLR import *
from numpy import inf, prod, all, sum
#from FuncDesigner.boundsurf import boundsurf

def r14IP(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, CBKPMV, g, nNodes,  \
         frc, fTol, Solutions, varTols, _in, dataType, \
         maxNodes, _s, indTC, xRecord):

    required_sigma = p.ftol * 0.99 # to suppress roundoff effects
    
    m, n = y.shape
    
    ip = func10(y, e, vv)
    ip.dictOfFixedFuncs = p.dictOfFixedFuncs
    ip.surf_preference = True

    tmp = asdf1.interval(ip, ia_surf_level=1)
#        print(type(tmp))
    if hasattr(tmp, 'resolve'):#type(tmp) == boundsurf:
#            print('b')
        #adjustr4WithDiscreteVariables(wr4, p)
        cs = oopoint((v, asarray(0.5*(val[0] + val[1]), dataType)) for v, val in ip.items())
        cs.dictOfFixedFuncs = p.dictOfFixedFuncs
        o, a = tmp.values(cs)
        definiteRange = tmp.definiteRange
    else:
        o, a, definiteRange = tmp.lb, tmp.ub, tmp.definiteRange

    
    if not all(definiteRange):
        p.err('''
        numerical integration with interalg is implemented 
        for definite (real) range only, no NaN values in integrand are allowed''')

    o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T

    nodes = func11(y, e, None, indTC, None, o, a, _s, p)

    an = nodes if len(_in) == 0 else hstack((_in, nodes)).tolist()
    
    if 1: 
        an.sort(key = lambda obj: obj.key, reverse=False)
        #an.sort(key = lambda obj: obj.minres, reverse=False)
    else:
        an.sort(key=lambda obj: obj.volumeResidual, reverse=False)

    ao_diff = array([node.key for node in an])
    volumes = array([node.volume for node in an])
    
    r10 = ao_diff <= 0.95*(required_sigma-p._residual) / (prod(p.ub-p.lb) - p._volume)
    ind = where(r10)[0]
    # TODO: use true_sum
    v = volumes[ind]
    p._F += sum(array([an[i].F for i in ind]) * v)
    residuals = ao_diff[ind] * v
    p._residual += residuals.sum()
    p._volume += v.sum()
    an = asarray(an, object)
    an = an[where(logical_not(r10))[0]]
        
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

    return an, g, inf, _s, Solutions, xRecord, frc, CBKPMV
