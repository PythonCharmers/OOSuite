import numpy
from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin, hstack, vstack, arange, amin, \
logical_and, float64, ceil, amax, inf, ndarray, isinf, any, logical_or, nan, take, logical_not, asanyarray, searchsorted, \
logical_xor
from numpy.linalg import norm, solve, LinAlgError
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F, MAX_NON_SUCCESS
from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point
from openopt.solvers.UkrOpt.interalgMisc import *
from FuncDesigner import sum as fd_sum, abs as fd_abs
   
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
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    iterfcnConnected = True
    fStart = None
    dataType = float64
    #maxMem = '150MB'
    maxNodes = 150000
    mn = 1500
    maxSolutions = 1
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()
    _requiresFiniteBoxBounds = True

    def __init__(self): pass
    def __solver__(self, p):
        if not p.__isFiniteBoxBounded__(): 
            p.err('solver %s requires finite lb, ub: lb <= x <= ub' % self.__name__)
#        if p.goal in ['max', 'maximum']:
#            p.err('solver %s cannot handle maximization problems yet' % self.__name__)
        if p.fixedVars is not None:
            p.err('solver %s cannot handle FuncDesigner problems with some variables declared as fixed' % self.__name__)
        if p.probType in ('LP', 'MILP', 'MINLP'):
            p.err("the solver can't handle problems of type " + p.probType)
        if not p.isFDmodel:
            p.err('solver %s can handle only FuncDesigner problems' % self.__name__)
        for val in p._x0.values():
            if isinstance(val,  (list, tuple, ndarray)) and len(val) > 1:
                p.err('''
                solver %s currently can handle only single-element variables, 
                use oovars(n) instead of oovar(size=n)'''% self.__name__)

        point = p.point
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X)
        p.kernelIterFuncs.pop(SMALL_DELTA_F)
        if MAX_NON_SUCCESS in p.kernelIterFuncs: 
            p.kernelIterFuncs.pop(MAX_NON_SUCCESS)
        
        if not bottleneck_is_present:
                p.pWarn('''
                installation of Python module "bottleneck" 
                (http://berkeleyanalytics.com/bottleneck,
                available via easy_install, takes several minutes for compilation)
                could speedup the solver %s''' % self.__name__)
        
        n = p.n
        
        # TODO: handle it in other level
        p.useMultiPoints = True
        
        maxSolutions = self.maxSolutions
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
        SolutionCoords = array([]).reshape(0, n)
        
        
        dataType = self.dataType
        if type(dataType) == str:
            if not hasattr(numpy, dataType):
                p.pWarn('your architecture has no type "%s", float64 will be used instead')
                dataType = 'float64'
            dataType = getattr(numpy, dataType)
        lb, ub = asarray(p.lb, dataType), asarray(p.ub, dataType)

        
        C = p.constraints
        vv = p._freeVarsList
        isSNLE = p.probType == 'NLSP'
        fTol = p.fTol
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
        
        # TODO: remove it after proper NLSP handling implementation
        if isSNLE:
            frc = 0.0
            eqs = [fd_abs(elem) for elem in p.user.f]
            asdf1 = fd_sum(eqs)
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

            tmp = asdf1(p._x0)
            if  tmp < frc:
                frc = tmp
                
            if p.fOpt is not None:
                if p.fOpt > frc:
                    p.warn('user-provided fOpt seems to be incorrect, ')
                frc = p.fOpt
        


#        if dataType==float64:
#            numBytes = 8 
#        elif self.dataType == 'float128':
#            numBytes = 16
#        else:
#            p.err('unknown data type, should be float64 or float128')
#        maxMem = self.maxMem
#        if type(maxMem) == str:
#            if maxMem.lower().endswith('kb'):
#                maxMem = int(float(maxMem[:-2]) * 2 ** 10)
#            elif maxMem.lower().endswith('mb'):
#                maxMem = int(float(maxMem[:-2]) * 2 ** 20)
#            elif maxMem.lower().endswith('gb'):
#                maxMem = int(float(maxMem[:-2]) * 2 ** 30)
#            elif maxMem.lower().endswith('tb'):
#                maxMem = int(float(maxMem[:-2]) * 2 ** 40)
#            else:
#                p.err('incorrect max memory parameter value, should end with KB, MB, GB or TB')
        m = 0
        #maxActive = 1
        self.mn = int(self.mn )
        self.maxNodes = int(self.maxNodes )

        _in = []
        
        y_excluded, e_excluded, o_excluded, a_excluded = [], [], [], []
        k = True
        g = inf
        C = p._FD.nonBoxCons
        isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__()
        
        # TODO: hanlde fixed variables here
        varTols = p.variableTolerances
        if maxSolutions != 1:
            if p.probType != 'NLSP':
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
        
        for itn in range(p.maxIter+10):
            ip = func10(y, e, vv)
            
#            for f, lb_, ub_ in C:
#                TMP = f.interval(domain, dataType)
#                lb, ub = asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType)
                
            o, a = func8(ip, asdf1, dataType)
            if p.debug and any(a + 1e-15 < o):  
                p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
            if p.debug and any(logical_xor(isnan(o), isnan(a))):
                p.err('bug in FuncDesigner intervals engine')
                
            FuncVals = getCentersValues(ip, asdf1, dataType) 

            xk, Min = getBestCenterAndObjective(FuncVals, ip, dataType)
            if CBKPMV > Min:
                CBKPMV = Min
                xRecord = xk# TODO: is copy required?
            if frc > Min:
                frc = Min
                
            if isSNLE:
                fo = 0.0#fTol / 16.0
            else:
                fo = min((frc, CBKPMV - (0.0 if maxSolutions == 1 else fTol))) 
            
            m = e.shape[0]
            o, a, FuncVals = o.reshape(2*n, m).T, a.reshape(2*n, m).T, FuncVals.reshape(2*n, m).T
            
            y, e, o, a, FuncVals = func7(y, e, o, a, FuncVals)

#            ind = all(e-y <= varTols, 1)
#            y_excluded += y[ind]
#            e_excluded += e[ind]
#            o_excluded += o[ind]
#            a_excluded += a[ind]
            
            if maxSolutions != 1:
                FCl, FCu = FuncVals[:, :n], FuncVals[:, n:]
                if isSNLE:
                    assert p.__isNoMoreThanBoxBounded__(), 'unimplemented yet'
                    candidates_L, candidates_U =  where(FCl < fTol), where(FCu < fTol)
                    Centers = 0.5 * (y + e)
                    Diff = 0.5 * (e - y)
                    candidates = []
                    # L
                    cs_L = Centers[candidates_L[0]].copy()
                    for I in range(len(candidates_L[0])):#TODO: rework it
                        i, j = candidates_L[0][I], candidates_L[1][I]
                        tmp = Centers[i].copy()
                        tmp[j] -= 0.5*Diff[i, j]
                        candidates.append(tmp)
                    # U
                    cs_U = Centers[candidates_U[0]].copy()
                    for I in range(len(candidates_U[0])):#TODO: rework it
                        i, j = candidates_U[0][I], candidates_U[1][I]
                        tmp = Centers[i].copy()
                        tmp[j] += 0.5*Diff[i, j]
                        candidates.append(tmp)
                    
                    for c in candidates:
                        ind = all(abs(c - SolutionCoords) < varTols, 1)
                        if not any(ind): 
                            solutions.append(c)
                            SolutionCoords = asarray(solutions, dataType)
                            
                    if len(solutions) >= maxSolutions:
                        k = False
                        solutions = solutions[:maxSolutions]
                        p.istop = 100
                        p.msg = 'required number of solutions has been obtained'
                        break
            
            p.iterfcn(xk, Min)
            if p.istop != 0: 
                break
            if isSNLE and maxSolutions == 1 and Min <= fTol:
                # TODO: rework it for nonlinear systems with non-bound constraints
                p.istop, p.msg = 1000, 'required solution has been obtained'
                break

            nodes = func11(y, e, o, a, FCl, FCu) if maxSolutions != 1 else func11(y, e, o, a)
            
#                ind = Fl < fTol
#                #tmp_solutions = nodes[ind]
#                lxs, uxs = y[ind], e[ind]
#                cs = 0.5*(lxs + uxs)
##                for s in solutions:
##                    pass
            
            # TODO: use sorted(..., key = lambda obj:obj.key) instead?
            # TODO: get rid of sorted, use get_n_min / get_n_max instead
            an = sorted(nodes + _in)
            
            an, g = func9(an, fo, g)


            # TODO: rework it
#            if len(an) == 0: 
#                k = False
#                p.istop, p.msg = 1000, 'optimal solution obtained'
#                break            

            nn = 1 if asdf1.isUncycled and all(isfinite(a)) and all(isfinite(o)) and isOnlyBoxBounded else self.maxNodes
            pnc = max((len(an), pnc))
            an, g = func5(an, nn, g)
            nNodes.append(len(an))
            
            y, e, _in = func12(an, self.mn, maxSolutions, solutions, SolutionCoords, varTols, fo)
            
            if y.size == 0: 
                k = False
                p.istop, p.msg = 1001, 'optimal solutions obtained'
                break
            nActiveNodes.append(y.shape[1])
            # End of main cycle
            
        p.iterfcn(xRecord)
        ff = p.fk # ff may be not assigned yet
        
        o = asarray([t.o for t in an])
        if o.size != 0:
            g = nanmin([nanmin(o), g])
        p.extras['isRequiredPrecisionReached'] = \
        True if ff - g < fTol and (k is False or isSNLE and maxSolutions==1) else False
        # TODO: simplify it
        if p.goal in ('max', 'maximum'):
            g = -g
            o = -o
        tmp = [nanmin(hstack((ff, g, o.flatten()))), numpy.asscalar(array((ff if p.goal not in ['max', 'maximum'] else -ff)))]
        if p.goal in ['max', 'maximum']: tmp = tmp[1], tmp[0]
        p.extras['extremumBounds'] = tmp
        
        p.solutions = [p._vector2point(s) for s in solutions]
        if p.iprint >= 0:
            s = 'Solution with required tolerance %0.1e \n is%s guarantied (obtained precision: %0.1e)' \
                   %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT', tmp[1]-tmp[0])
            if not p.extras['isRequiredPrecisionReached'] and pnc == self.maxNodes: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)


    

    

