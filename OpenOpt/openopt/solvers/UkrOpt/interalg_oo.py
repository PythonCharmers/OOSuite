from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin, hstack, vstack, tile, arange, amin, \
logical_and, float64, ceil, amax, inf, ndarray
import numpy
from numpy.linalg import norm, solve, LinAlgError
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point
from FuncDesigner import ooPoint

class interalg(baseSolver):
    __name__ = 'interalg_0.15'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    iterfcnConnected = True
    fStart = None
    dataType = float64
    maxNodes = 15000
    maxActiveNodes = 1500
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()

    def __init__(self): pass
    def __solver__(self, p):
        if not p.__isFiniteBoxBounded__(): 
            p.err('solver %s requires finite lb, ub: lb <= x <= ub' % self.__name__)
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
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X)
        p.kernelIterFuncs.pop(SMALL_DELTA_F)
        p.useMultiPoints = True
        
        nN = []        
        p.extras['nNodes'] = nN
        nAn = []
        p.extras['nActiveNodes'] = nAn
        
        dataType = self.dataType
        if type(dataType) == str:
            if not hasattr(numpy, dataType):
                p.pWarn('your architecture has no type "%s", float64 will be used instead')
                dataType = 'float64'
            dataType = getattr(numpy, dataType)
        lb, ub = asarray(p.lb, dataType), asarray(p.ub, dataType)

        n = p.n
        f = p.f
        fTol = p.fTol
        ooVars = p._freeVarsList
        
        fd_obj = p.user.f[0]
        if p.goal in ('max', 'maximum'):
            fd_obj = -fd_obj

        xRecord = 0.5 * (lb + ub)

        BestKnownMinValue = p.f(xRecord)    
        y = lb.reshape(1, -1)
        u = ub.reshape(1, -1)
        fRecord = inf
        
        fStart = self.fStart

        if fStart is not None and fStart < BestKnownMinValue: 
            fRecord = fStart
        tmp = fd_obj(p._x0)
        if  tmp < fRecord:
            fRecord = tmp
        if p.fOpt is not None:
            if p.fOpt > fRecord:
                p.err('user-provided fOpt seems to be incorrect')
            fRecord = p.fOpt
        
        def getIntervals(y, u):
            LB = [[] for i in range(n)]
            UB = [[] for i in range(n)]

            for i in range(n):
                lb, ub = y[:, i], u[:, i]
                center = 0.5 * (lb + ub) 
                LB[i] = hstack((tile(lb, n), tile(lb, i), center, tile(lb, n-i-1)))
                UB[i] = hstack((tile(ub, i), center, tile(ub, n-i-1), tile(ub, n)))

            d = dict([(v, (LB[i], UB[i])) for i, v in enumerate(ooVars)])
            
            d = ooPoint(d, skipArrayCast = True)
            d.isMultiPoint = True
            TMP = fd_obj.interval(d)
            
            a = dict([(key, 0.5*(val[0]+val[1])) for key, val in d.items()])
            a = ooPoint(a, skipArrayCast = True)
            a.isMultiPoint = True
            F = fd_obj(a)
            bCInd = argmin(F)

            bC = array([0.5*(val[0][bCInd]+val[1][bCInd]) for val in d.values()])
            bCObjective = atleast_1d(F)[bCInd]
            return asarray(TMP.lb), asarray(TMP.ub), bC, bCObjective


        m = 0
        maxActiveNodes = self.maxActiveNodes 
        if fd_obj.isUncycled: 
            maxActiveNodes = 1
            maxNodes = 1
        y_i = array([]).reshape(0, n)
        u_i = array([]).reshape(0, n)
        o_i = array([]).reshape(0, 2*n)
        e_i = array([]).reshape(0, 2*n)
        PointsLeft = True
        cutLevel = inf
        
        for itn in range(p.maxIter+10):
            o, e, bC, bCObjective = getIntervals(y, u)
            xk, Min = bC, bCObjective
            p.iterfcn(xk, Min)
            if p.istop != 0 : return
            if BestKnownMinValue > Min:
                BestKnownMinValue = Min
                xRecord = xk
            if fRecord > BestKnownMinValue:
                fRecord = BestKnownMinValue 
            if fTol is None:
                fTol = 1e-7
                p.warn('solver %s require p.fTol value (required objective function tolerance); 10^-7 will be used')
            th = min((fRecord, BestKnownMinValue - fTol)) 
            m = u.shape[0]
            o, e = o.reshape(2*n, m).T, e.reshape(2*n, m).T
            a = 0.5*(y + u)
            o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
            for i in range(n):
                ind = where(o_modL[:, i] > th)[0]
                if ind.size != 0:
                    y[:,i][ind] = a[:,i][ind]
                ind = where(o_modU[:, i] > th)[0]
                if ind.size != 0:
                    u[:,i][ind] = a[:,i][ind]
            y, u, o, e = vstack((y, y_i)), vstack((u, u_i)), vstack((o, o_i)), vstack((e, e_i))
            setForRemoving = set()
            o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
            for i in range(n):
                ind = where(logical_and(o_modL[:, i] > th, o_modU[:, i] > th))[0]
                if ind.size != 0:
                    setForRemoving.update(ind.tolist())
                    cutLevel = amin((cutLevel, amin(o_modL[ind, i]), amin(o_modU[ind, i])))
            if len(setForRemoving) != 0:
                ind = array(list(setForRemoving))
                o_modL, o_modU = delete(o_modL, ind, 0), delete(o_modU, ind, 0)
                y, u, o, e = delete(y, ind, 0), delete(u, ind, 0), delete(o, ind, 0), delete(e, ind, 0)
            if u.size == 0: 
                PointsLeft = False
                p.istop = 1000
                p.msg = 'optimal solution obtained'
                break            
            m = u.shape[0]
            o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
            if m > self.maxNodes:
                numOfElementsToBeRemoved = m - self.maxNodes
                tmp = where(o_modU<o_modL, o_modU, o_modL)
                ind = argmax(tmp, 1)
                values = tmp[arange(m),ind]
                ind = values.argsort()
                indCut = m-numOfElementsToBeRemoved-1
                cutLevel = amin((values[indCut], cutLevel))
                ind = ind[m-numOfElementsToBeRemoved:]
                o_modL, o_modU = delete(o_modL, ind, 0), delete(o_modU, ind, 0)
                y, u, o, e = delete(y, ind, 0), delete(u, ind, 0), delete(o, ind, 0), delete(e, ind, 0)
            m = y.shape[0]
            if m > maxActiveNodes:
                o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
                tmp = where(o_modU<o_modL, o_modU, o_modL)
                ind = argmax(tmp, 1)
                values = tmp[arange(m),ind]
                ind = values.argsort()
                ind = ind[maxActiveNodes:]
                y_i, u_i, o_i, e_i = y[ind], u[ind], o[ind], e[ind]
                y, u, o, e = delete(y, ind, 0), delete(u, ind, 0), delete(o, ind, 0), delete(e, ind, 0)
            m = y.shape[0]
            nAn.append(m)
            nN.append(m + y_i.shape[0])
            am = arange(m)
            bcfs = argmin(e, 1) % n
            new_y, new_u = y.copy(), u.copy()
            nC = 0.5 * (new_y[am, bcfs] + new_u[am, bcfs])
            new_y[am, bcfs] = nC
            new_u[am, bcfs] = nC
            new_y = vstack((y, new_y))
            new_u = vstack((new_u, u))
            u, y = new_u, new_y
        ff = f(xRecord)
        p.iterfcn(xRecord, ff)
        p.extras['isRequiredPrecisionReached'] = True if ff - cutLevel < fTol and PointsLeft is False else False
        if p.goal in ('max', 'maximum'):
            cutLevel = -cutLevel
            o = -o
        tmp = [amin(hstack((ff, cutLevel, o.flatten()))), numpy.asscalar(array((ff if p.goal not in ['max', 'maximum'] else -ff)))]
        if p.goal in ['max', 'maximum']: tmp = tmp[1], tmp[0]
        p.extras['extremumBounds'] = tmp
        if p.iprint >= 0:
            s = 'Solution with required tolerance %0.1e is%s guarantied (obtained precision: %0.3e)' \
                   %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT', tmp[1]-tmp[0])
            p.info(s)
