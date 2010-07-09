#from numpy import asarray,  ones, all, isfinite, copy, nan, concatenate, array, dot
#from openopt.kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from openopt.kernel.setDefaultIterFuncs import SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON,  SMALL_DELTA_X, SMALL_DELTA_F
from numpy import isfinite, asscalar, asfarray, abs, copy, isinf, ones, array
import nlopt

def NLOPT_AUX(p, solver):
    
    if p.nb > 0 or p.nbeq > 0:
        p.err('general linear constraints are not implemented yet')
    
    def myfunc(x, grad):
        #if p.istop != 0: raise nlopt.FORCED_STOP
        if grad.size > 0:
            grad[:]= p.df(x.copy())
        return asscalar(p.f(x))
        #return p.f(x)
    
    opt = nlopt.opt(solver, p.n)
    opt.set_min_objective(myfunc)
    if any(p.lb==p.ub): p.pWarn('nlopt solvers badly handle problems with variables fixed via setting lb=ub')
    lb = [elem if isfinite(elem) else float(elem) for elem in p.lb.tolist()]
    ub = [elem if isfinite(elem) else float(elem) for elem in p.ub.tolist()]
    if any(isfinite(lb)): opt.set_lower_bounds(lb)
    if any(isfinite(ub)): opt.set_upper_bounds(ub)
    
    # TODO: A, Aeq
    
    if p.nc > 0:
        def c(result, x, grad):
            if grad.size > 0:
                grad[:] = p.dc(x.copy())
            result[:] = p.c(x)
        opt.add_inequality_mconstraint(c, array([p.contol]*p.nc))
        
    if p.nb > 0:
        def c_lin(result, x, grad):
            if grad.size > 0:
                grad[:] = p.A.copy()
            result[:] = p.__get_AX_Less_B_residuals__(x)
        opt.add_inequality_mconstraint(c_lin, array([p.contol]*p.nc))
    
    if p.nh > 0:
        def h(result, x, grad):
            if grad.size > 0:
                tmp = p.dh(x.copy())
                if hasattr(tmp, 'toarray'): tmp = tmp.toarray()
                grad[:] = tmp
            result[:] = p.h(x).copy()
        #opt.add_equality_mconstraint(h, array([0.75*p.contol]*p.nh))
        
    
    if isfinite(p.maxTime): 
        opt.set_maxtime(p.maxTime)
        
#    opt.set_xtol_rel(1e-1)
#    opt.set_ftol_rel(1e-1)

    opt.set_xtol_abs(p.xtol)
    opt.set_ftol_abs(p.ftol)
    opt.set_maxeval(p.maxFunEvals)
    # others like fEnough, maxFunEvals, are handled by OO  kernel
    
    x0 = asfarray(p.x0).copy()
    
    #lb2 = copy(lb)
    #lb2[isinf(lb2)] = 0
    LB = copy(lb) #+ 1e-15*(ones(p.n) + abs(lb2))
    ind = x0 <= LB
    x0[ind] = LB[ind]
        
    #ub2 = copy(ub)
    #ub2[isinf(ub2)] = 0
    UB = copy(ub) #- 1e-15*(ones(p.n) + abs(ub2))
    ind = x0 >= UB
    x0[ind] = UB[ind]
    
    
    #x0[ind] = p.lb[ind] + 1e-15*abs((max(1.0, p.lb[x0<p.lb])))
    #x0[x0>p.ub] = p.ub[x0>p.ub] - 1e-15*min((1.0, abs(p.ub[x0>p.ub])))
    
    x = opt.optimize(x0.tolist())
    if p.point(p.xk).betterThan(p.point(x)):
        x = p.xk
        #p.iterfcn(x)
    p.xk = x
    
    iStop = opt.get_stopval()

    if p.istop == 0:
        if iStop == nlopt.XTOL_REACHED:
            p.istop,  p.msg = SMALL_DELTA_X, '|| X[k] - X[k-1] || < xtol'
        elif iStop == nlopt.FTOL_REACHED:
            p.istop,  p.msg = SMALL_DELTA_F, '|| F[k] - F[k-1] || < ftol'
        else:
            p.istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
    
    #raise 0
