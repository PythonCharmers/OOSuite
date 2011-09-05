import numpy
from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin, hstack, vstack, arange, amin, \
logical_and, float64, ceil, amax, inf, ndarray, isinf, any, logical_or, nan, take, logical_not, asanyarray, searchsorted, \
logical_xor, empty
from numpy.linalg import norm, solve, LinAlgError
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F, MAX_NON_SUCCESS, IS_NAN_IN_X
from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point
from openopt.solvers.UkrOpt.interalgMisc import *
from FuncDesigner import sum as fd_sum, abs as fd_abs, max as fd_max
from ii_engine import *
from interalgCons import processConstraints

bottleneck_is_present = False
try:
    from bottleneck import nanargmin, nanargmax, nanmin
    bottleneck_is_present = True
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax



class interalg(baseSolver):
    __name__ = 'interalg_0.21'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'c', 'h', 'A', 'Aeq', 'b', 'beq']
    iterfcnConnected = True
    fStart = None
    dataType = float64
    #maxMem = '150MB'
    maxNodes = 150000
    maxActiveNodes = 150
    dataHandling = 'auto'
    _requiresBestPointDetection = True
    
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__() #and p.probType != 'IP'
    _requiresFiniteBoxBounds = True

    def __init__(self): pass
    def __solver__(self, p):
        if not p.__isFiniteBoxBounded__(): 
            p.err('''
            solver %s requires finite lb, ub: 
            lb <= x <= ub 
            (you can use "implicitBoounds")
            ''' % self.__name__)
        if p.fixedVars is not None:
            p.err('solver %s cannot handle FuncDesigner problems with some variables declared as fixed' % self.__name__)
        if p.probType in ('LP', 'MILP', 'MINLP'):
            p.err("the solver can't handle problems of type " + p.probType)
        if not p.isFDmodel:
            p.err('solver %s can handle only FuncDesigner problems' % self.__name__)
        
        isOpt = p.probType in ['NLP', 'NSP', 'GLP']
        
        isSNLE = p.probType == 'NLSP'
        if not p.__isNoMoreThanBoxBounded__ and not isSNLE:
            p.warn('handling constraints by the solver interalg is undone properly yet for all problems except of SNLE')
        
        isIP = p.probType == 'IP'
        if isIP:
            pb = r14IP
            p._F = asarray(0, self.dataType)
            p._residual = 0.0
            f_int = p.user.f[0].interval(p.domain, self.dataType)
            p._r0 = prod(p.ub-p.lb) * (f_int.ub - f_int.lb)
            p._volume = 0.0
            p.kernelIterFuncs.pop(IS_NAN_IN_X)
        else:
            pb = r14
        
        for val in p._x0.values():
            if isinstance(val,  (list, tuple, ndarray)) and len(val) > 1:
                p.err('''
                solver %s currently can handle only single-element variables, 
                use oovars(n) instead of oovar(size=n)'''% self.__name__)

        point = p.point
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        p.kernelIterFuncs.pop(MAX_NON_SUCCESS, None)
        
        if not bottleneck_is_present:
                p.pWarn('''
                installation of Python module "bottleneck" 
                (http://berkeleyanalytics.com/bottleneck,
                available via easy_install, takes several minutes for compilation)
                could speedup the solver %s''' % self.__name__)
        
#        if isSNLE and not p.__isNoMoreThanBoxBounded__():
#           p.err('constrained systems of equations are unimplemented yet')
        
        n = p.n
        
        # TODO: handle it in other level
        p.useMultiPoints = True
        
        maxSolutions = p.maxSolutions
        if maxSolutions == 0: maxSolutions = 10**50
        if maxSolutions != 1 and p.fEnough != -inf:
            p.warn('''
            using the solver interalg with non-single solutions mode 
            is not ajusted with fEnough stop criterium yet, it will be omitted
            ''')
            p.kernelIterFuncs.pop(FVAL_IS_ENOUGH)
            
        
        nNodes = []        
        p.extras['nNodes'] = nNodes
        nActiveNodes = []
        p.extras['nActiveNodes'] = nActiveNodes

        solutions = []
        r6 = array([]).reshape(0, n)
        
        dataType = self.dataType
        if type(dataType) == str:
            if not hasattr(numpy, dataType):
                p.pWarn('your architecture has no type "%s", float64 will be used instead')
                dataType = 'float64'
            dataType = getattr(numpy, dataType)

        lb, ub = asarray(p.lb, dataType), asarray(p.ub, dataType)

        vv = list(p._freeVarsList)

        fTol = p.fTol
        if isIP:
            if p.ftol is None:
                if fTol is not None:
                    p.ftol = fTol
                else:
                    p.err('for intergration problems interalg requires user-supplied ftol (required precision)')
            if fTol is None: fTol = p.ftol
            elif fTol != p.ftol:
                p.err('you have provided both ftol and fTol for intergation problem')

        if fTol is None:
            fTol = 1e-7
            p.warn('solver %s require p.fTol value (required objective function tolerance); 10^-7 will be used' % self.__name__)

        xRecord = 0.5 * (lb + ub)

        CBKPMV = inf
        
        y = lb.reshape(1, -1)
        e = ub.reshape(1, -1)
        frc = inf

        # TODO: maybe rework it, especially for constrained case
        fStart = self.fStart
        
        # TODO: remove it after proper SNLE handling implementation
        if isSNLE:
            frc = 0.0
            eqs = [fd_abs(elem) for elem in p.user.f]
            asdf1 = fd_sum(eqs)
            
            # TODO: check it, for reducing calculations
            #C.update([elem == 0 for elem in p.user.f])
        else:
            asdf1 = p.user.f[0]
            
            if p.fOpt is not None:  fOpt = p.fOpt
            if p.goal in ('max', 'maximum'):
                asdf1 = -asdf1
                if p.fOpt is not None:
                    fOpt = -p.fOpt
            
            if fStart is not None and fStart < CBKPMV: 
                frc = fStart
                
            for X0 in [point(xRecord), point(p.x0)]:
                if X0.isFeas(altLinInEq=False) and X0.f() < CBKPMV:
                    CBKPMV = X0.f()

            if p.isFeas(p.x0):
                tmp = asdf1(p._x0)
                if  tmp < frc:
                    frc = tmp
                
            if p.fOpt is not None:
                if p.fOpt > frc:
                    p.warn('user-provided fOpt seems to be incorrect, ')
                frc = p.fOpt

        if isSNLE:
            if self.dataHandling == 'raw':
                p.pWarn('''
                    this interalg data handling approach ("%s") 
                    is unimplemented for SNLE yet, dropping to "sorted"'''%self.dataHandling)
            
            # handles 'auto' as well
            self.dataHandling ='sorted'

        domain = dict([(v, [p.lb[i], p.ub[i]]) for i,  v in enumerate(vv)])
        
        if self.dataHandling == 'auto':
            if isIP:
                self.dataHandling = 'sorted'
            else:
                M = 0
                for func in [p.user.f[0]] + [Elem[0] for Elem in p._FD.nonBoxCons]:
                    r = func.interval(domain, self.dataType)
                    M = max((M, max(atleast_1d(abs(r.lb)))))
                    M = max((M, max(atleast_1d(abs(r.ub)))))
                self.dataHandling = 'raw' if M < 1e5 else 'sorted'
                #print M
                    
            #self.dataHandling = 'sorted' if isIP or (p.__isNoMoreThanBoxBounded__() and n < 50) else 'raw'
            
        p._isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__() 
        if asdf1.isUncycled and p._isOnlyBoxBounded and all(isfinite(p.user.f[0].interval(domain).lb)):
            #maxNodes = 1
            self.dataHandling = 'sorted'
            

        self.maxActiveNodes = int(self.maxActiveNodes)
#        if self.maxActiveNodes < 2:
#            p.warn('maxActiveNodes should be at least 2 while you have provided %d. Setting it to 2.' % self.maxActiveNodes)
        self.maxNodes = int(self.maxNodes)

        _in = array([], object)
        
        g = inf
        C = p._FD.nonBoxConsWithTolShift
        C0 = p._FD.nonBoxCons
        if isOpt:
            r = []
            for (elem, lb, ub, tol) in C0:
                if tol == 0: tol = p.contol
                if lb == ub:
                    r.append(fd_max((fd_abs(elem-lb)-tol, 0)) * (fTol/tol))
                elif lb == -inf:
                    r.append(fd_max((0, elem-ub-tol)) * (fTol/tol))
                elif ub == inf:
                    r.append(fd_max((0, lb-elem-tol)) * (fTol/tol))
                else:
                    p.err('finite box constraints are unimplemented for interalg yet')
            #p._cons_obj = 1e100 * fd_sum(r) if len(r) != 0 else None
            p._cons_obj = fd_sum(r) if len(r) != 0 else None
        if isSNLE:
            p._cons_obj = None # TODO: check it!
            C += [(elem, -(elem.tol if elem.tol != 0 else p.ftol), (elem.tol if elem.tol != 0 else p.ftol)) for elem in p.user.f]
            C0 += [(elem, 0, 0, (elem.tol if elem.tol != 0 else p.ftol)) for elem in p.user.f]
        
        
        
        # TODO: hanlde fixed variables here
        varTols = p.variableTolerances
        if maxSolutions != 1:
            if not isSNLE:
                p.err('''
                "search several solutions" mode is unimplemented
                for the prob type %s yet''' % p.probType)
            if any(varTols == 0):
                p.err('''
                for the mode "search all solutions" 
                you have to provide all non-zero tolerances 
                for each variable (oovar)
                ''')
            
        pnc = 0
        an = []
        maxNodes = self.maxNodes
        _s = nan
        
        for itn in range(p.maxIter+10):
            if len(C0) != 0: 
                ip = func10(y, e, vv)
                m = e.shape[0]
                y, e, nlhc = processConstraints(C0, y, e, ip, m, p, dataType)
            else:
                nlhc = None
            
            if y.size != 0:
                an, g, fo, _s, solutions, r6, xRecord, frc, CBKPMV = \
                pb(p, nlhc, y, e, vv, asdf1, C, CBKPMV, itn, g, \
                             nNodes, frc, fTol, maxSolutions, varTols, solutions, r6, _in, \
                             dataType, maxNodes, _s, xRecord)
                if _s is None:
                    break
            else:
                an = _in
                fo = 0.0 if isSNLE else min((frc, CBKPMV - (fTol if maxSolutions == 1 else 0.0))) 
            pnc = max((len(an), pnc))
            
            if isIP:
                y, e, _in, _s = \
                    func12(an, self.maxActiveNodes, p, solutions, r6, vv, varTols, inf, 'IP')# Case=3
            else:
                y, e, _in, _s = \
                func12(an, self.maxActiveNodes, p, solutions, r6, vv, varTols, fo, 2) # Case=2
            nActiveNodes.append(y.shape[0]/2)
            if y.size == 0: 
                if len(solutions) > 1:
                    p.istop, p.msg = 1001, 'all solutions have been obtained'
                else:
                    p.istop, p.msg = 1000, 'solution has been obtained'
                break            
            ############# End of main cycle ###############
            
        if not isSNLE and not isIP:
            if p._bestPoint.betterThan(p.point(p.xk)):
                p.iterfcn(p._bestPoint)
            else:
                p.iterfcn(p.xk)
            
            
        ff = p.fk # ff may be not assigned yet
#        ff = p._bestPoint.f()
#        p.xk = p._bestPoint.x
        if isIP: 
            p.xk = array([nan]*p.n)
            p.rk = p._residual
            p.fk = p._F
            
        isFeas = p.isFeas(p.xk) if not isIP else p.rk < fTol
        if not isFeas and p.istop > 0:
            p.istop, p.msg = -1000, 'no feasible solution has been obtained'
        
        o = asarray([t.o for t in an])
        if o.size != 0:
            g = nanmin([nanmin(o), g])
            
        p.extras['isRequiredPrecisionReached'] = \
        True if ff - g < fTol and isFeas else False
        # and (k is False or (isSNLE and (p._nObtainedSolutions >= maxSolutions or maxSolutions==1))) 
        
        if not p.extras['isRequiredPrecisionReached'] and p.istop > 0:
            p.istop = -1
            p.msg = 'required precision is not guarantied'
        # TODO: simplify it
        if p.goal in ('max', 'maximum'):
            g = -g
            o = -o
        tmp = [nanmin(hstack((ff, g, o.flatten()))), numpy.asscalar(array((ff if p.goal not in ['max', 'maximum'] else -ff)))]
        if p.goal in ['max', 'maximum']: tmp = tmp[1], tmp[0]
        p.extras['extremumBounds'] = tmp if not isIP else 'unimplemented for IP yet'
        
        p.solutions = [p._vector2point(s) for s in solutions]
        if p.maxSolutions == 1: delattr(p, 'solutions')
        if p.iprint >= 0:
#            s = 'Solution with required tolerance %0.1e \n is%s guarantied (obtained precision: %0.1e)' \
#                   %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT', tmp[1]-tmp[0])
            s = 'Solution with required tolerance %0.1e \n is%s guarantied' \
            %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT')
            if not isIP and p.maxSolutions == 1:
                s += ' (obtained precision: %0.1e)' % (tmp[1]-tmp[0])
            if not p.extras['isRequiredPrecisionReached'] and pnc == self.maxNodes: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)


    

    

