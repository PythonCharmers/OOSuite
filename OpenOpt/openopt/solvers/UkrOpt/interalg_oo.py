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
        f = p.f
        C = p.constraints
        ooVars = p._freeVarsList
        
        fTol = p.fTol
        if fTol is None:
            fTol = 1e-7
            p.warn('solver %s require p.fTol value (required objective function tolerance); 10^-7 will be used' % self.__name__)

        xRecord = 0.5 * (lb + ub)

        CurrentBestKnownPointsMinValue = inf
            
        Lx = lb.reshape(1, -1)
        Ux = ub.reshape(1, -1)
        fRecord = inf

        fd_obj = p.user.f[0]
        if p.fOpt is not None:  fOpt = p.fOpt
        if p.goal in ('max', 'maximum'):
            fd_obj = -fd_obj
            if p.fOpt is not None:
                fOpt = -p.fOpt

        
        # TODO: maybe rework it, especially for constrained case
        fStart = self.fStart

        if fStart is not None and fStart < CurrentBestKnownPointsMinValue: 
            fRecord = fStart
            
        for X0 in [point(xRecord), point(p.x0)]:
            if X0.isFeas(altLinInEq=False) and X0.f() < CurrentBestKnownPointsMinValue:
                CurrentBestKnownPointsMinValue = X0.f()
            
        tmp = fd_obj(p._x0)
        if  tmp < fRecord:
            fRecord = tmp
            
        if p.fOpt is not None:
            if fOpt > fRecord:
                p.warn('user-provided fOpt seems to be incorrect, ')
            fRecord = fOpt
        


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
        
        Lx_inactive = array([]).reshape(0, n)
        Ux_inactive = array([]).reshape(0, n)
        maxLf_inactive = []
        inactiveNodes = []
        Lf_inactive = array([]).reshape(0, 2*n)
        Uf_inactive = array([]).reshape(0, 2*n)
        
        Lx_excluded, Ux_excluded, Lf_excluded, Uf_excluded = [], [], [], []
        PointsLeft = True
        cutLevel = inf
        C = p._FD.nonBoxCons
        isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__()
        varTols = p.variableTolerances
        
        
        for itn in range(p.maxIter+10):
            ip = formIntervalPoint(Lx, Ux, n, ooVars)
            
#            for f, lb_, ub_ in C:
#                TMP = f.interval(domain, dataType)
#                lb, ub = asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType)
                
            Lf, Uf, bestCenter, bestCenterObjective = getIntervals(ip, fd_obj, dataType)
            
            if p.debug and any(Uf<Lf):  p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
            if p.debug and any(logical_xor(isnan(Lf), isnan(Uf))):
                p.err('bug in FuncDesigner intervals engine')
            
            xk, Min = bestCenter, bestCenterObjective
           
            p.iterfcn(xk, Min)
            if p.istop != 0: 
                break
            
            if CurrentBestKnownPointsMinValue > Min:
                CurrentBestKnownPointsMinValue = Min
                xRecord = xk# TODO: is copy required?
            if fRecord > Min:
                fRecord = Min

            threshold = min((fRecord, CurrentBestKnownPointsMinValue - (0.0 if self.allSolutions else fTol))) 
            
            m = Ux.shape[0]
            Lf, Uf = Lf.reshape(2*n, m).T, Uf.reshape(2*n, m).T
            Lx, Ux, Lf, Uf = remove_NaN_nodes(Lx, Ux, Lf, Uf)
            m = Ux.shape[0]
            Lf_modL, Lf_modU = Lf[:, 0:n], Lf[:, n:2*n]
            tmp = where(Lf_modU<Lf_modL, Lf_modU, Lf_modL)
            ind = nanargmax(tmp, 1)
            maxLf = tmp[arange(m),ind]


#            ind = all(Ux-Lx <= varTols, 1)
#            Lx_excluded += Lx[ind]
#            Ux_excluded += Ux[ind]
#            Lf_excluded += Lf[ind]
#            Uf_excluded += Uf[ind]

            nodes = formNodes(Lx, Ux, Lf, Uf, maxLf)
            
            # TODO: use sorted(..., key = lambda obj:obj.key) instead?
            AllNodes = sorted(chain(nodes, inactiveNodes))
            
            AllNodes, cutLevel = removeSomeNodes(AllNodes, n, threshold, cutLevel)


            # TODO: rework it
            if len(AllNodes) == 0: 
                # For more safety against bugs
                activeNodes = []
                inactiveNodes = []
                
                PointsLeft = False
                p.istop, p.msg = 1000, 'optimal solution obtained'
                break            

            nCut = 1 if fd_obj.isUncycled and all(isfinite(Uf)) and all(isfinite(Lf)) and isOnlyBoxBounded else self.maxNodes
            AllNodes, cutLevel = TruncateOutOfAllowedNumberNodes(AllNodes, n, nCut, cutLevel)
            
            activeNodes, inactiveNodes = makeSomeNodesInactive(AllNodes, n, self.maxActiveNodes)

            m = len(activeNodes)
            nActiveNodes.append(m)
            
            nNodes.append(len(activeNodes) + len(inactiveNodes))

            Lx, Ux, Lf, Uf = asanyarray([t.data[0] for t in activeNodes]), \
            asanyarray([t.data[1] for t in activeNodes]), \
            asanyarray([t.data[2] for t in activeNodes]), \
            asanyarray([t.data[3] for t in activeNodes])
            
            Lx, Ux = TruncateSomeBoxes(Lx, Ux, Lf, Uf, n, threshold)
            
            bestCoordsForSplitting = getBestCoordsForSplitting(Lx, Ux, Lf, Uf, n, varTols)
            Lx, Ux = formNewBoxes(Lx, Ux, bestCoordsForSplitting)
            # End of main cycle
            
        ff = f(xRecord)
        p.iterfcn(xRecord, ff)
        
        Lf = asarray([t.data[2] for t in AllNodes])
        if Lf.size != 0:
            cutLevel = nanmin([nanmin(Lf), cutLevel])
        p.extras['isRequiredPrecisionReached'] = True if ff - cutLevel < fTol and PointsLeft is False else False
        # TODO: simplify it
        if p.goal in ('max', 'maximum'):
            cutLevel = -cutLevel
            Lf = -Lf
        tmp = [nanmin(hstack((ff, cutLevel, Lf.flatten()))), numpy.asscalar(array((ff if p.goal not in ['max', 'maximum'] else -ff)))]
        if p.goal in ['max', 'maximum']: tmp = tmp[1], tmp[0]
        p.extras['extremumBounds'] = tmp
        if p.iprint >= 0:
            s = 'Solution with required tolerance %0.1e \n is%s guarantied (obtained precision: %0.1e)' \
                   %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT', tmp[1]-tmp[0])
            if not p.extras['isRequiredPrecisionReached']: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)


    

    

