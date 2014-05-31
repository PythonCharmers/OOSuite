from numpy import empty, logical_and, logical_not, take, zeros, isfinite, any, \
asarray, ndarray, bool_#where
from interalgT import adjustDiscreteVarBounds, truncateByPlane
import numpy as np

# for PyPy
from openopt.kernel.nonOptMisc import where

hasPoint = lambda y, e, point:\
    True if y.size != 0 and any([(np.all(y[i]<=point) and np.all(e[i]>=point)) for i in range(y.shape[0])]) else False
pointInd = lambda y, e, point:\
    where([(np.all(y[i]<=point) and np.all(e[i]>=point)) for i in range(y.shape[0])])[0].tolist()
    
def processConstraints(C0, y, e, _s, p, dataType):
    #P = np.array([  7.64673334e-01,    4.35551807e-01,    5.93869991e+02,   5.00000000e+00])
#    P = np.array([-0.63521194458007812, -0.3106536865234375, 0.0905609130859375, 0.001522064208984375, -0.69999999999999996, -0.99993896484375, 0.90000152587890625, 1.0, 4.0])

#    print('c-1', p.iter, hasPoint(y, e, P), pointInd(y, e, P))
    n = p.n
    m = y.shape[0]
    indT = empty(m, bool)
    indT.fill(False)
#    isSNLE = p.probType in ('NLSP', 'SNLE')
    
    for i in range(p.nb):
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, p.A[i], p.b[i]+p.contol)
        if ind_trunc is not True:
            _s = _s[ind_trunc]
    for i in range(p.nbeq):
        # TODO: handle it via one func
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, p.Aeq[i], p.beq[i]+p.contol)
        if ind_trunc is not True:
            _s = _s[ind_trunc]
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, -p.Aeq[i], -p.beq[i]+p.contol)
        if ind_trunc is not True:
            _s = _s[ind_trunc]
   
    
    DefiniteRange = True
    if len(p._discreteVarsNumList):
        y, e, _s, indT = adjustDiscreteVarBounds(y, e, _s, indT, p)

    m = y.shape[0]
    nlh = zeros((m, 2*n))
    nlh_0 = zeros(m)
    fullOutput = False#isSNLE and not p.hasLogicalConstraints
    residual = zeros((m, 2*n)) if fullOutput else None
    residual_0 = zeros(m) if fullOutput else None
    
    for itn, (c, f, lb, ub, tol) in enumerate(C0):
#        print ('c_1', itn, c.dep, hasPoint(y, e, P))
        m = y.shape[0] # is changed in the cycle
        if m == 0: 
            return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), residual, True, False, _s
            #return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), residual.reshape(0, 2*n), True, False, _s
        assert nlh.shape[0] == y.shape[0]
        
        if fullOutput:
            (T0, Res0), (res, R_res), DefiniteRange2 = c.nlh(y, e, p, dataType, fullOutput = True)
            residual_0 += Res0
        else:
            # may be logical constraint and doesn't have kw fullOutput at all
            T0, res, DefiniteRange2 = c.nlh(y, e, p, dataType)
        DefiniteRange = logical_and(DefiniteRange, DefiniteRange2)
        
        assert T0.ndim <= 1, 'unimplemented yet'

        nlh_0 += T0
        assert nlh.shape[0] == m
        # TODO: rework it for case len(p._freeVarsList) >> 1

        for v, tmp in res.items():
            j = p._freeVarsDict.get(v)
            nlh[:, n+j] += tmp[:, tmp.shape[1]/2:].flatten() - T0
            nlh[:, j] += tmp[:, :tmp.shape[1]/2].flatten() - T0
            if fullOutput:
                Tmp = R_res[v]
                residual[:, n+j] += Tmp[:, Tmp.shape[1]/2:].flatten() - Res0
                residual[:, j] += Tmp[:, :Tmp.shape[1]/2].flatten() - Res0
                    
        assert nlh.shape[0] == m
        ind = where(logical_and(any(isfinite(nlh), 1), isfinite(nlh_0)))[0]
        lj = ind.size
        if lj != m:
            y = take(y, ind, axis=0, out=y[:lj])
            e = take(e, ind, axis=0, out=e[:lj])
            nlh = take(nlh, ind, axis=0, out=nlh[:lj])
            nlh_0 = nlh_0[ind]
#            residual = take(residual, ind, axis=0, out=residual[:lj])
            indT = indT[ind]
            _s = _s[ind]
            if fullOutput:
                residual_0 = residual_0[ind]
                residual = take(residual, ind, axis=0, out=residual[:lj])
            if asarray(DefiniteRange).size != 1: 
                DefiniteRange = take(DefiniteRange, ind, axis=0, out=DefiniteRange[:lj])
#            print ('c_2', itn, c.dep, hasPoint(y, e, P))
        assert nlh.shape[0] == y.shape[0]


        ind = logical_not(isfinite(nlh))
        if any(ind):
            indT[any(ind, 1)] = True
            
            ind_l,  ind_u = ind[:, :ind.shape[1]/2], ind[:, ind.shape[1]/2:]
            tmp_l, tmp_u = 0.5 * (y[ind_l] + e[ind_l]), 0.5 * (y[ind_u] + e[ind_u])
            y[ind_l], e[ind_u] = tmp_l, tmp_u
            # TODO: mb implement it
            if len(p._discreteVarsNumList):
                if tmp_l.ndim > 1:
                    # shouldn't reduce y,e shape here, so output values doesn't matter
                    l1 = len(_s)
                    tmp_l, tmp_u, _s2, indT2 = adjustDiscreteVarBounds(tmp_l, tmp_u, _s, indT, p)
                    l2 = len(_s2)
                    if l1 != l2:
                        print('Warning: possible bug in interalg constraints processing, inform developers')
                else:
                    y, e, _s, indT = adjustDiscreteVarBounds(y, e, _s, indT, p)

            nlh_l, nlh_u = nlh[:, nlh.shape[1]/2:], nlh[:, :nlh.shape[1]/2]
            
            # copy() is used because += and -= operators are involved on nlh in this cycle and probably some other computations
            nlh_l[ind_u], nlh_u[ind_l] = nlh_u[ind_u].copy(), nlh_l[ind_l].copy()   
            if fullOutput:
                residual_l, residual_u = residual[:, residual.shape[1]/2:], residual[:, :residual.shape[1]/2]
                residual_l[ind_u], residual_u[ind_l] = residual_u[ind_u].copy(), residual_l[ind_l].copy()   
#            print ('c_3', itn, c.dep, hasPoint(y, e, P))

    if nlh.size != 0:
        if DefiniteRange is False:
            nlh_0 += 1e-300
        elif type(DefiniteRange) == ndarray and not all(DefiniteRange):
            nlh_0[logical_not(DefiniteRange)] += 1e-300
        else:
            assert type(DefiniteRange) in (bool, bool_, ndarray)
    # !! matrix - vector
    nlh += nlh_0.reshape(-1, 1)
    
    if fullOutput:
        # !! matrix - vector
        residual += residual_0.reshape(-1, 1)
        residual[residual_0>=1e300] = 1e300
    
    #print('c2', p.iter, hasPoint(y, e, P), pointInd(y, e, P))
    return y, e, nlh, residual, DefiniteRange, indT, _s

hasPoint = lambda y, e, point:\
    True if y.size != 0 and any([(all(y[i]<=point) and all(e[i]>=point)) for i in range(y.shape[0])]) else False
pointInd = lambda y, e, point:\
    where([(all(y[i]<=point) and all(e[i]>=point)) for i in range(y.shape[0])])[0].tolist()
    


