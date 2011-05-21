from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin, hstack, vstack, arange, amin, \
logical_and, float64, ceil, amax, inf, ndarray, isinf, any, logical_or, nan, take, logical_not, asanyarray, searchsorted

import numpy
from numpy.linalg import norm, solve, LinAlgError
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F, MAX_NON_SUCCESS
from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point
from openopt.solvers.UkrOpt.interalgMisc import *

hasHeapMerge = False
try:
    from heapq import merge
    hasHeapMerge = True
except ImportError:
    pass

try:
    from bottleneck import nanargmin, nanargmax, nanmin
    bottleneck_is_present = True
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax
    bottleneck_is_present = False
#from numpy import nanmin, nanargmin, nanargmax

class interalg(baseSolver):
    __name__ = 'interalg_0.17'
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
    useArrays4Store = False
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()
    #_canHandleScipySparse = True

    #lv default parameters

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

        if not hasHeapMerge:
            p.pWarn('''cannot import merge from heapq, 
            maybe you using Python version prior to 2.6
            useArrays4Store will be set to False
            (this is slower and more unstable mode, 
            maybe even buggy, out of maintanance)
            ''')
            self.useArrays4Store = True
        

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
        f = p.f
        C = p.constraints
        ooVars = p._freeVarsList
        
        fTol = p.fTol
        if fTol is None:
            fTol = 1e-7
            p.warn('solver %s require p.fTol value (required objective function tolerance); 10^-7 will be used' % self.__name__)

        xRecord = 0.5 * (lb + ub)

        CurrentBestKnownPointsMinValue = inf
            
        y = lb.reshape(1, -1)
        e = ub.reshape(1, -1)
        fr = inf

        fd_obj = p.user.f[0]
        if p.fOpt is not None:  fOpt = p.fOpt
        if p.goal in ('max', 'maximum'):
            fd_obj = -fd_obj
            if p.fOpt is not None:
                fOpt = -p.fOpt

        
        # TODO: maybe rework it, especially for constrained case
        fStart = self.fStart

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
        
        b = array([]).reshape(0, n)
        e_inactive = array([]).reshape(0, n)
        maxo_inactive = []
        _in = []
        o_inactive = array([]).reshape(0, 2*n)
        a_inactive = array([]).reshape(0, 2*n)
        
        y_excluded, e_excluded, o_excluded, a_excluded = [], [], [], []
        k = True
        g = inf
        C = p._FD.nonBoxCons
        isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__()
        varTols = p.variableTolerances
        nCut = 1 if fd_obj.isUncycled and all(isfinite(a)) and all(isfinite(o)) and isOnlyBoxBounded else self.maxNodes
        
        for itn in range(p.maxIter+10):
            ip = func10(y, e, n, ooVars)
            
#            for f, lb_, ub_ in C:
#                TMP = f.interval(domain, dataType)
#                lb, ub = asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType)
                
            o, a, bestCenter, bestCenterObjective = func8(ip, fd_obj, dataType)
            
            if p.debug and any(a<o):  p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
            
            xk, Min = bestCenter, bestCenterObjective
           
            p.iterfcn(xk, Min)
            if p.istop != 0: 
                break
            
            if CurrentBestKnownPointsMinValue > Min:
                CurrentBestKnownPointsMinValue = Min
                xRecord = xk# TODO: is copy required?
            if fr > Min:
                fr = Min

            fo = min((fr, CurrentBestKnownPointsMinValue - fTol)) 
            
            m = e.shape[0]
            o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T
            y, e, o, a = func7(y, e, o, a)
            m = e.shape[0]
            o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
            tmp = where(o_modU<o_modL, o_modU, o_modL)
            ind = nanargmax(tmp, 1)
            maxo = tmp[arange(m),ind]

            if self.useArrays4Store:
                y, e, o, a, maxo = \
                vstack((y, b)), vstack((e, e_inactive)), vstack((o, o_inactive)), vstack((a, a_inactive)), hstack((maxo, maxo_inactive))

                y, e, o, a, maxo, g = func6(y, e, o, a, maxo, n, fo, g)
                y, e, o, a, maxo, g = func5(y, e, o, a, maxo,  n, nCut, g)
                y, e, o, a, maxo, b, e_inactive, o_inactive, a_inactive, maxo_inactive =\
                func3(y, e, o, a, maxo, n, self.maxActiveNodes)
                m = y.shape[0]
                nActiveNodes.append(m)
                nNodes.append(m + b.shape[0])

#            ind = all(e-y <= varTols, 1)
#            y_excluded += y[ind]
#            e_excluded += e[ind]
#            o_excluded += o[ind]
#            a_excluded += a[ind]
            else:
                nodes = func11(y, e, o, a, maxo)
                nodes.sort()
                an = list(merge(nodes, _in))
                an, g = func9(an, n, fo, g)
            

            # TODO: rework it
            if (self.useArrays4Store and y.size == 0) or (not self.useArrays4Store and len(an) == 0): 
                if len(an) == 0 and not self.useArrays4Store: 
                    # For more safety against bugs
                    an1 = []
                    _in = []
                k = False
                p.istop, p.msg = 1000, 'optimal solution obtained'
                break            
            
            
            
            if not self.useArrays4Store:
                an, g = func52(an, n, nCut, g)
                
                an1, _in = func32(an, n, self.maxActiveNodes)

                m = len(an1)
                nActiveNodes.append(m)
                
                nNodes.append(len(an1) + len(_in))

                y, e, o, a = asanyarray([t.data[0] for t in an1]), \
                asanyarray([t.data[1] for t in an1]), \
                asanyarray([t.data[2] for t in an1]), \
                asanyarray([t.data[3] for t in an1])
            
            y, e = func4(y, e, o, a, n, fo)
            
            t = func1(y, e, o, a, n, varTols)
            y, e = func2(y, e, t)
            # End of main cycle
            
        ff = f(xRecord)
        p.iterfcn(xRecord, ff)
        if not self.useArrays4Store:
            o = asanyarray([t.data[2] for t in an])
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
            if not p.extras['isRequiredPrecisionReached']: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)


    

    

