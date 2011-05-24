import numpy
from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin, hstack, vstack, arange, amin, \
logical_and, float64, ceil, amax, inf, ndarray, isinf, any, logical_or, nan, take, logical_not, asanyarray, searchsorted, \
logical_xor
from numpy.linalg import norm, solve, LinAlgError
from itertools import chain
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
    maxActiveNodes = 1500
    allSolutions = False
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
        
        # TODO: handle it in other level
        p.useMultiPoints = True
        
        nNodes = []        
        p.extras['nNodes'] = nNodes
        nActiveNodes = []
        p.extras['nActiveNodes'] = nActiveNodes
        
        dataType = self.dataType
        if type(dataType) == str:
            if not hasattr(numpy, dataType):
                p.pWarn('your architecture has no type "%s", float64 will be used instead')
                dataType = 'float64'
            dataType = getattr(numpy, dataType)
        lb, ub = asarray(p.lb, dataType), asarray(p.ub, dataType)

        n = p.n
        C = p.constraints
        ooVars = p._freeVarsList
        
        if p.probType == 'NLSP':
            fTol = p.ftol
            if p.fTol is not None:
                fTol = min((p.ftol, p.fTol))
                p.warn('''
                both ftol and fTol are passed to the NLSP;
                minimal value of the pair will be used (%0.1e);
                also, you can modify each personal tolerance for equation, e.g. 
                equations = [(sin(x)+cos(y)=-0.5)(tol = 0.001), ...]
                ''' % fTol)
        else:
            fTol = p.fTol
            if fTol is None:
                fTol = 1e-7
                p.warn('solver %s require p.fTol value (required objective function tolerance); 10^-7 will be used' % self.__name__)

        xRecord = 0.5 * (lb + ub)

        CurrentBestKnownPointsMinValue = inf
        
        y = lb.reshape(1, -1)
        e = ub.reshape(1, -1)
        fr = inf

        # TODO: maybe rework it, especially for constrained case
        fStart = self.fStart
        
        # TODO: remove it after proper NLSP handling implementation
        if p.probType == 'NLSP':
            fr = 0.0
            eqs = [fd_abs(elem) for elem in p.user.f]
            fd_obj = fd_sum(eqs)
        else:
            fd_obj = p.user.f[0]
            
            if p.fOpt is not None:  fOpt = p.fOpt
            if p.goal in ('max', 'maximum'):
                fd_obj = -fd_obj
                if p.fOpt is not None:
                    fOpt = -p.fOpt
            
                
            if fStart is not None and fStart < CurrentBestKnownPointsMinValue: 
                fr = fStart
                
            for X0 in [point(xRecord), point(p.x0)]:
                if X0.isFeas(altLinInEq=False) and X0.f() < CurrentBestKnownPointsMinValue:
                    CurrentBestKnownPointsMinValue = X0.f()

            tmp = fd_obj(p._x0)
            if  tmp < fr:
                fr = tmp
                
            if p.fOpt is not None:
                if fOpt > fr:
                    p.warn('user-provided fOpt seems to be incorrect, ')
                fr = fOpt
        


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
        #maxActiveNodes = self.maxActiveNodes 

        _in = []
        
        y_excluded, e_excluded, o_excluded, a_excluded = [], [], [], []
        k = True
        g = inf
        C = p._FD.nonBoxCons
        isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__()
        varTols = p.variableTolerances
        peak_nodes_number = 0
        
        
        for itn in range(p.maxIter+10):
            ip = func10(y, e, ooVars)
            
#            for f, lb_, ub_ in C:
#                TMP = f.interval(domain, dataType)
#                lb, ub = asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType)
                
            o, a, bestCenter, bestCenterObjective = func8(ip, fd_obj, dataType)
            #print nanmin(o), nanmin(a)
            if p.debug and any(a + 1e-15 < o):  
                p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
            if p.debug and any(logical_xor(isnan(o), isnan(a))):
                p.err('bug in FuncDesigner intervals engine')
            
            xk, Min = bestCenter, bestCenterObjective
           
            p.iterfcn(xk, Min)
            if p.istop != 0: 
                break
            
            if CurrentBestKnownPointsMinValue > Min:
                CurrentBestKnownPointsMinValue = Min
                xRecord = xk# TODO: is copy required?
            if fr > Min:
                fr = Min

            fo = min((fr, CurrentBestKnownPointsMinValue - (0.0 if self.allSolutions else fTol))) 
            
            m = e.shape[0]
            o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T
            y, e, o, a = func7(y, e, o, a)
            m = e.shape[0]
            o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
            tmp = where(o_modU<o_modL, o_modU, o_modL)
            ind = nanargmax(tmp, 1)
            maxo = tmp[arange(m),ind]


#            ind = all(e-y <= varTols, 1)
#            y_excluded += y[ind]
#            e_excluded += e[ind]
#            o_excluded += o[ind]
#            a_excluded += a[ind]

            nodes = func11(y, e, o, a, maxo)
            
            # TODO: use sorted(..., key = lambda obj:obj.key) instead?
            an = sorted(chain(nodes, _in))
            
            an, g = func9(an, fo, g)


            # TODO: rework it
            if len(an) == 0: 
                # For more safety against bugs
                an1 = []
                _in = []
                
                k = False
                p.istop, p.msg = 1000, 'optimal solution obtained'
                break            

            nCut = 1 if fd_obj.isUncycled and all(isfinite(a)) and all(isfinite(o)) and isOnlyBoxBounded else self.maxNodes
            peak_nodes_number = max((len(an), peak_nodes_number))
            an, g = func5(an, nCut, g)
            
            an1, _in = func3(an, self.maxActiveNodes)

            m = len(an1)
            nActiveNodes.append(m)
            
            nNodes.append(len(an1) + len(_in))

            y, e, o, a = asanyarray([t.data[0] for t in an1]), \
            asanyarray([t.data[1] for t in an1]), \
            asanyarray([t.data[2] for t in an1]), \
            asanyarray([t.data[3] for t in an1])
            
            y, e = func4(y, e, o, a, n, fo)
            t = func1(y, e, o, a, varTols)
            y, e = func2(y, e, t)
            # End of main cycle
            
        p.iterfcn(xRecord)
        ff = p.fk # ff may be not assigned yet
        
        o = asarray([t.data[2] for t in an])
        if o.size != 0:
            g = nanmin([nanmin(o), g])
        p.extras['isRequiredPrecisionReached'] = True if ff - g < fTol and k is False else False
        # TODO: simplify it
        if p.goal in ('max', 'maximum'):
            g = -g
            o = -o
        tmp = [nanmin(hstack((ff, g, o.flatten()))), numpy.asscalar(array((ff if p.goal not in ['max', 'maximum'] else -ff)))]
        if p.goal in ['max', 'maximum']: tmp = tmp[1], tmp[0]
        p.extras['extremumBounds'] = tmp
        if p.iprint >= 0:
            s = 'Solution with required tolerance %0.1e \n is%s guarantied (obtained precision: %0.1e)' \
                   %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT', tmp[1]-tmp[0])
            if not p.extras['isRequiredPrecisionReached'] and peak_nodes_number == self.maxNodes: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)


    

    

