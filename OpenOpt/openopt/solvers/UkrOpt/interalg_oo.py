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
    mn = 150
    
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
#        if self.mn < 2:
#            p.warn('mn should be at least 2 while you have provided %d. Setting it to 2.' % self.mn)
        self.maxNodes = int(self.maxNodes )

        _in = []#array([], object)
        
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
#                o, a = func8(ip, asdf1, dataType)
#                m = o.size/(2*n)
#                o, a  = o.reshape(2*n, m).T, a.reshape(2*n, m).T
#                lf1, lf2, uf1, uf2 = o[:, 0:n], o[:, n:2*n], a[:, 0:n], a[:, n:2*n]
#                o, a = nanmax(where(lf1>lf2, lf2, lf1), 1), nanmin(where(uf1>uf2, uf1, uf2), 1)
#                
#                # TODO: add tol?
#                ind = logical_or(a < _lb, o > _ub)
                
            o, a = func8(ip, asdf1, dataType)
            if p.debug and any(a + 1e-15 < o):  
                p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
            if p.debug and any(logical_xor(isnan(o), isnan(a))):
                p.err('bug in FuncDesigner intervals engine')
                
            r3 = getr4Values(ip, asdf1, dataType) 

            xk, Min = r2(r3, ip, dataType)
            
            if CBKPMV > Min:
                CBKPMV = Min
                xRecord = xk# TODO: is copy required?
            if frc > Min:
                frc = Min
            
            fo = 0.0 if isSNLE else min((frc, CBKPMV - (fTol if maxSolutions == 1 else 0.0))) 
            
            m = e.shape[0]
            o, a, r3 = o.reshape(2*n, m).T, a.reshape(2*n, m).T, r3.reshape(2*n, m).T
            if itn == 0: 
                _s = atleast_1d(nanmax(a))
            y, e, o, a, r3, _s = func7(y, e, o, a, r3, _s)

#            ind = all(e-y <= varTols, 1)
#            y_excluded += y[ind]
#            e_excluded += e[ind]
#            o_excluded += o[ind]
#            a_excluded += a[ind]
            
            if maxSolutions != 1:
                FCl, FCu = r3[:, :n], r3[:, n:]
                if isSNLE:
                    assert p.__isNoMoreThanBoxBounded__(), 'unimplemented yet'
                    r5_L, r5_U =  where(FCl < fTol), where(FCu < fTol)
                    r4 = 0.5 * (y + e)
                    Diff = 0.5 * (e - y)
                    r5 = []
                    # L
                    cs_L = r4[r5_L[0]].copy()
                    for I in range(len(r5_L[0])):#TODO: rework it
                        i, j = r5_L[0][I], r5_L[1][I]
                        tmp = r4[i].copy()
                        tmp[j] -= 0.5*Diff[i, j]
                        r5.append(tmp)
                    # U
                    cs_U = r4[r5_U[0]].copy()
                    for I in range(len(r5_U[0])):#TODO: rework it
                        i, j = r5_U[0][I], r5_U[1][I]
                        tmp = r4[i].copy()
                        tmp[j] += 0.5*Diff[i, j]
                        r5.append(tmp)
                    
                    for c in r5:
                        ind = all(abs(c - r6) < varTols, 1)
                        if not any(ind): 
                            solutions.append(c)
                            r6 = asarray(solutions, dataType)
                            
                    p._nObtainedSolutions = len(solutions)
                    
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
            
            nodes = func11(y, e, o, a, _s, FCl, FCu) if maxSolutions != 1 else func11(y, e, o, a, _s)
            
#                ind = Fl < fTol
#                #tmp_solutions = nodes[ind]
#                lxs, uxs = y[ind], e[ind]
#                cs = 0.5*(lxs + uxs)
##                for s in solutions:
##                    pass
            
            # TODO: get rid of sorted, use get_n_min / get_n_max instead
            an = sorted(nodes + _in, key = lambda obj: obj.key)
            
#            nodes.sort(key = lambda obj: obj.key)
#            an = [nodes.pop(0) if len(nodes)!=0 and (len(_in)==0 or nodes[0].key< _in[0].key)\
#                                                          else _in.pop(0) for i in range(len(nodes) + len(_in))]

            #assert all([an[i] is an2[i] for i in range(len(an))])
#            arr1 = [node.key for node in _in]
#            arr2 = [node.key for node in nodes]
#            from numpy import searchsorted, insert
#            r10 = searchsorted(arr1, arr2)
#            if _in == []: _in = array([], object)
#            an = insert(_in, r10, nodes)

#            an = nodes + _in
#            arr_n = array([node.key for node in an])
#            I = arr_n.argsort()
#            an = [an[i] for i in I]
            
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
            
            y, e, _in, _s = \
            func12(an, self.mn, maxSolutions, solutions, r6, varTols, fo, _s)
            nActiveNodes.append(y.shape[0]/2)
            if y.size == 0: 
                k = False
                if len(solutions) > 1:
                    p.istop, p.msg = 1001, 'optimal solutions obtained'
                else:
                    p.istop, p.msg = 1000, 'optimal solution obtained'
                break            
            
            # End of main cycle
            
        p.iterfcn(xRecord)
        ff = p.fk # ff may be not assigned yet
        
        o = asarray([t.o for t in an])
        if o.size != 0:
            g = nanmin([nanmin(o), g])
        p.extras['isRequiredPrecisionReached'] = \
        True if ff - g < fTol and (k is False or isSNLE and maxSolutions==1) else False
        if not p.extras['isRequiredPrecisionReached'] and p.istop > 0:
            p.istop = -1
            p.msg = 'required precision is not guarantied'
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


    

    

