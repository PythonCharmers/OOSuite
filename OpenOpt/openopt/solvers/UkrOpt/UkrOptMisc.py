from numpy import concatenate, vstack, hstack, inf, ones, zeros, diag, dot, abs, where, max, min
from openopt import NLP, LLSP
from numpy.linalg import norm
from openopt.kernel.setDefaultIterFuncs import IS_MAX_ITER_REACHED

funcDirectionValue = lambda alpha, func, x, direction:  func(x+alpha*direction)

def funcDirectionValueWithMaxConstraintLimit(alpha, func, x, direction, maxConstrLimit, p):
    xx = x+alpha*direction
    mc = p.getMaxResidual(xx)
    if mc > maxConstrLimit:
        return 1e150 * mc
    else:
        return func(xx)
#    cond1 = not(p_LS is None) and hasattr(p_LS, 'xPrevF') and alpha1 == p_LS.xPrevF
#
#    if cond1:
#        constr1 = p.getMaxResidual(x+alpha1*direction)
#        constr2 = p.getMaxResidual(x+alpha2*direction)
#    else:
#        constr2 = p.getMaxResidual(x+alpha2*direction)
#        constr1 = p.getMaxResidual(x+alpha1*direction)
#    if constr2 > constr1 and constr2 > maxConstrLimit: return True
#    elif constr1 > constr2 and constr1 > maxConstrLimit: return False
#
#    if cond1:
#        f1 = p.f(x+alpha1*direction)
#        f2 = p.f(x+alpha2*direction)
#    else:
#        f2 = p.f(x+alpha2*direction)
#        f1 = p.f(x+alpha1*direction)
#    if f1 < f2: return True
#    else: return False

def getConstrDirection(p,  x, regularization = 1e-7):
    c, dc, h, dh, df = p.c(x), p.dc(x), p.h(x), p.dh(x), p.df(x)
    A, Aeq = vstack((dc, p.A)), vstack((dh, p.Aeq))
    b = concatenate((-c, p.b-p.matmult(p.A,x)))
    beq = concatenate((-h, p.beq-p.matmult(p.Aeq,x)))
    lb = p.lb - x
    ub = p.ub - x

    nC = b.size

    #TODO: USE if nC == 0 instead
    if A.size == 0: A_LLS = hstack((zeros((nC, p.n)), diag(ones(nC))))
    else: A_LLS = hstack((A, diag(ones(nC))))

    if Aeq.size == 0:
        Awhole_LLS = vstack((A_LLS, hstack((diag(regularization*ones(p.n)), zeros((p.n, nC))))))
    else:
        Aeq_LLS = hstack((Aeq, zeros((beq.size, nC))))
        Awhole_LLS = vstack((Aeq_LLS, A_LLS, hstack((diag(regularization*ones(p.n)), zeros((p.n, nC))))))

    if p.getMaxResidual(x) > p.contol: dump_x = -p.getMaxConstrGradient(x)
    else: dump_x = -df

    bwhole_LLS = concatenate((beq, b, dump_x))
    lb_LLS = hstack((lb, zeros(nC)))
    ub_LLS = hstack((ub, inf*ones(nC)))
    p_sp = LLSP(Awhole_LLS,  bwhole_LLS,  lb = lb_LLS, ub = ub_LLS, iprint = -1)
    r_sp = p_sp.solve('BVLS', BVLS_inf=1e30)
    return r_sp.xf[:p.n]

class DirectionOptimPoint:
    def __init__(self):
        pass

def getDirectionOptimPoint(p, func, x, direction, forwardMultiplier = 2.0, maxiter = 150, xtol = None, maxConstrLimit = None,  \
                           alpha_lb = 0,  alpha_ub = inf,  \
                           rightLocalization = 0,  leftLocalization = 0, \
                           rightBorderForLocalization = 0, leftBorderForLocalization = None):
    if all(direction==0): p.err('nonzero direction is required')

    if maxConstrLimit is None:
        lsFunc = funcDirectionValue
        args = (func, x, direction)
    else:
        lsFunc = funcDirectionValueWithMaxConstraintLimit
        args = (func, x, direction, maxConstrLimit, p)

    prev_alpha, new_alpha = alpha_lb, min(alpha_lb+0.5, alpha_ub)
    prev_val = lsFunc(prev_alpha, *args)
    for i in xrange(p.maxLineSearch):
        if lsFunc(new_alpha, *args)>prev_val or new_alpha==alpha_ub: break
        else:
            if i != 0: prev_alpha = min(alpha_lb, new_alpha)
            new_alpha *= forwardMultiplier

    if i == p.maxLineSearch-1: p.debugmsg('getDirectionOptimPoint: maxLineSearch is exeeded')
    lb, ub = prev_alpha, new_alpha

    if xtol is None: xtol = p.xtol / 2.0
    # NB! goldenSection solver ignores x0
    p_LS = NLP(lsFunc, x0=0, lb = lb,  ub = ub, iprint = -1, \
               args=args, xtol = xtol, maxIter = maxiter, contol = p.contol)# contol is used in funcDirectionValueWithMaxConstraintLimit


    r = p_LS.solve('goldenSection', useOOiterfcn=False, rightLocalization=rightLocalization, leftLocalization=leftLocalization, rightBorderForLocalization=rightBorderForLocalization, leftBorderForLocalization=leftBorderForLocalization)
    if r.istop == IS_MAX_ITER_REACHED:
        p.warn('getDirectionOptimPoint: max iter has been exceeded')
    alpha_opt = r.special.rightXOptBorder

    R = DirectionOptimPoint()
    R.leftAlphaOptBorder = r.special.leftXOptBorder
    R.leftXOptBorder = x + R.leftAlphaOptBorder * direction
    R.rightAlphaOptBorder = r.special.rightXOptBorder
    R.rightXOptBorder = x + R.rightAlphaOptBorder * direction
    R.x = x + alpha_opt * direction
    R.alpha = alpha_opt
    R.evalsF = r.evals['f']+i
    return R

from numpy import arccos

def getAltitudeDirection(p, pointTop, point2, point3):
    b = point2 - pointTop
    c = point3 - pointTop
    alpha = dot(c, c-b) / norm(c-b)**2
    h = alpha * b + (1-alpha) * c

    abh = arccos(dot(h, b)/norm(h) /norm(b))
    ach = arccos(dot(h, c)/norm(h) /norm(c))
    abc = arccos(dot(b, c)/norm(b) /norm(c))
    #ahbc = arccos(dot(h,  b-c)/norm(h) /norm(b-c))
    isInsideTriangle = abh+ach-abc<=1e-8
    return h, isInsideTriangle

from openopt.kernel.setDefaultIterFuncs import IS_LINE_SEARCH_FAILED
from openopt.kernel.ooMisc import isSolved

def getBestPointAfterTurn(oldPoint, newPoint, altLinInEq=None, maxLS = 3, maxDeltaX = None, maxDeltaF = None, \
                          line_points = None, hs=None, new_bs = True):
    assert altLinInEq is not None
    p = oldPoint.p
    contol = p.contol

    c1, lin_eq1, lin_ineq1, lb1, ub1 = oldPoint.c(), oldPoint.lin_eq(), oldPoint.lin_ineq(), oldPoint.lb(), oldPoint.ub()

    c2, lin_eq2, lin_ineq2, lb2, ub2 = newPoint.c(), newPoint.lin_eq(), newPoint.lin_ineq(), newPoint.lb(), newPoint.ub()
    
    altPoint = p.point((oldPoint.x + newPoint.x) / 2.0)
    altPoint._lin_eq = (lin_eq1 + lin_eq2) / 2.0
    altPoint._lin_ineq = (lin_ineq1 + lin_ineq2) / 2.0

    ind1 = c1 > 0
    ind2 = c2 > 0
    ind = where(ind1 | ind2)[0]
    
    _c = zeros(p.nc)
    if ind.size != 0:
        _c[ind] = p.c((oldPoint.x + newPoint.x) / 2.0, ind)

    altPoint._c = _c



    if line_points is not None:
        pv = 0.5*hs
        line_points[pv] = altPoint.f()

    
    if maxLS is None:
        maxLS = p.maxLineSearch
        
    assert maxLS != 0
    
    if maxDeltaX is None: 
        if p.iter != 0:
            maxDeltaX = 1.5e4*p.xtol
        else:
            maxDeltaX = 1.5e1*p.xtol
    if maxDeltaF is None: maxDeltaF = 150*p.ftol
    prev_prev_point = newPoint
    for ls in xrange(maxLS):
        
        altPoint, prevPoint = p.point((oldPoint.x + altPoint.x) / 2.0), altPoint

        if line_points is not None:
            pv /= 2.0
            line_points[pv] = altPoint.f()

        c2, lin_eq2, lin_ineq2 = prevPoint._c, prevPoint._lin_eq, prevPoint._lin_ineq
        lin_eq2, lin_ineq2 = prevPoint._lin_eq, prevPoint._lin_ineq
        ind2 = c2 > 0
        ind = where(ind1 | ind2)[0]
        _c = zeros(p.nc)
        if ind.size != 0: _c[ind] = p.c((oldPoint.x + prevPoint.x) / 2.0, ind)
        altPoint._c = _c
        altPoint._lin_eq = (lin_eq1 + lin_eq2) / 2.0
        altPoint._lin_ineq = (lin_ineq1 + lin_ineq2) / 2.0


        #!!! "not betterThan" is used vs "betterThan" because prevPoint can become same to altPoint
        if not altPoint.betterThan(prevPoint, altLinInEq=altLinInEq):
            if new_bs:
                bestPoint, pointForDilation = prevPoint, prev_prev_point
                return bestPoint, pointForDilation, -1-ls
            else:
                return prev_prev_point, -1-ls
        prev_prev_point = prevPoint

        if p.norm(oldPoint.x - altPoint.x) < maxDeltaX: 
            #print 'insufficient step norm during backward search:', p.norm(oldPoint.x - altPoint.x) , maxDeltaX
            break
            
        rr = oldPoint.f() - altPoint.f()
        if (oldPoint.isFeas() or altPoint.isFeas()) and maxDeltaF/2**15 < abs(rr) < maxDeltaF: break
        
        # TODO: is it worth to implement?
        if (oldPoint.mr() > contol and altPoint.mr() > p.contol) and altPoint.mr() > 0.9*oldPoint.mr(): break 

    # got here if maxLS is exeeded; altPoint is better than prevPoint
    if new_bs:
        bestPoint, pointForDilation = altPoint, prevPoint
        return bestPoint, pointForDilation, -1-ls
    else:
        return prev_prev_point, -1-ls
    #return prev_prev_point, -ls







