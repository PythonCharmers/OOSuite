from numpy import diag, array, sqrt,  eye, ones, inf, any, copy, zeros, dot, where, all, tile, sum, nan, isfinite, float64, isnan, log10, \
max, sign, array_equal, nonzero, ix_, arctan, pi, logical_not, logical_and, atleast_2d, matrix, delete
from numpy.linalg import norm, solve, LinAlgError
#try:
#    from numpy.linalg import cond
#except:
#    print 'warning: no cond in numpy.linalg, matrix B rejuvenation check will be omitted'
#    cond = lambda Matrix: 1

from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point
from openopt.kernel.ooMisc import economyMult, Len
from openopt.kernel.setDefaultIterFuncs import *
from UkrOptMisc import getBestPointAfterTurn
from PolytopProjection import PolytopProjection

class gsubg(baseSolver):
    __name__ = 'gsubg'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Nikolay G. Zhurbenko generalized epsilon-subgradient"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    iterfcnConnected = True
    _canHandleScipySparse = True

    #gsubg default parameters
    h0 = 1.0
    hmult = 0.5
    T = float64
    
    showLS = False
    show_hs = False
    showRes = False
    show_nnan = False
    doBackwardSearch = True
    new_bs = True
    approach = 'all active'
    zhurb = 50
    sigma = 1e-3
    dual = True
    ls_direction = 'socp'

    def __init__(self): pass
    def __solver__(self, p):

        h0 = self.h0

        T = self.T
        # alternatively instead of alp=self.alp etc you can use directly self.alp etc

        n = p.n
        x0 = p.x0
        
        if p.nbeq == 0 or any(abs(p._get_AeqX_eq_Beq_residuals(x0))>p.contol): # TODO: add "or Aeqconstraints(x0) out of contol"
            x0[x0<p.lb] = p.lb[x0<p.lb]
            x0[x0>p.ub] = p.ub[x0>p.ub]
        
        hs = asarray(h0, T)
        ls_arr = []

        """                         Nikolay G. Zhurbenko generalized epsilon-subgradient engine                           """
        bestPoint = p.point(asarray(copy(x0), T))
        bestFeasiblePoint = None if not bestPoint.isFeas(True) else bestPoint
        prevIter_best_ls_point = bestPoint
        best_ls_point = bestPoint
        prevIter_bestPointAfterTurn = bestPoint
        bestPointBeforeTurn = bestPoint
        g = bestPoint._getDirection(self.approach)
        if not any(g) and all(isfinite(g)):
            # TODO: create ENUMs
            p.istop = 14 if bestPoint.isFeas(False) else -14
            p.msg = 'move direction has all-zero coords'
            return

        HS = []
        LS = []
        
        # TODO: add possibility to handle f_opt if known instead of Ftol
        Ftol = p.Ftol/2.0 if hasattr(p, 'Ftol') else 15 * p.ftol
        
        objGradVectors, vectorNorms, points, values, isConstraint, usefulness, inactive = [], [], [], [], [], [], []
        StoredInfo = [objGradVectors, vectorNorms, points, values, isConstraint, usefulness, inactive]
        nVec = self.zhurb
        
        """                           gsubg main cycle                                    """

        for itn in xrange(1500000):
            
            iterStartPoint = prevIter_best_ls_point
            x = iterStartPoint.x.copy()

            schedule = [bestPoint] if itn == 0 else [point for point in (bestPointAfterTurn, bestPointBeforeTurn) if id(point.x) != id(points[-1])]
#            if itn != 0:
#                x = (bestPointAfterTurn.x+bestPointBeforeTurn.x) / 2.0
#                iterStartPoint = p.point(x)

            
            nAddedVectors = 0
            for point in schedule:
                if isfinite(point.f()) and bestFeasiblePoint is not None:
                    tmp = point.df()
                    n_tmp = norm(tmp)
                    if n_tmp < p.gtol:
                        p._df = n_tmp # TODO: change it 
                        p.iterfcn(point)
                        return
                    objGradVectors.append(tmp)
                    vectorNorms.append(n_tmp)
                    val = point.f()
                    values.append(asscalar(val))
                    usefulness.append(asscalar(val  - dot(point.x, tmp))/n_tmp)
                    isConstraint.append(False)
                    points.append(point.x)
                    inactive.append(0)
                    nAddedVectors += 1
                if not point.isFeas(True):
                    # TODO: use old-style w/o the arg "currBestFeasPoint = bestFeasiblePoint"
                    #tmp = point._getDirection(self.approach, currBestFeasPoint = bestFeasiblePoint)
                    tmp = point._getDirection(self.approach)
                    n_tmp = norm(tmp)
                    objGradVectors.append(tmp)
                    vectorNorms.append(n_tmp)
                    val = point.mr_alt()
                    values.append(val)
                    usefulness.append(asscalar(val  - dot(point.x, tmp))/n_tmp)
                    isConstraint.append(True)
                    points.append(point.x)
                    inactive.append(0)
                    nAddedVectors += 1
                    
            indToBeRemoved = []
            
            if len(values) > nAddedVectors:
                for j in range(nAddedVectors):
                    ind = -1-j
                    #vectors, norms = asarray(objGradVectors[:ind]), asarray(vectorNorms[:ind])
                    vectors, norms = asarray(objGradVectors), asarray(vectorNorms)
                    scalarProducts = dot(vectors, objGradVectors[ind])
                    normProducts = norms * vectorNorms[ind]
                    IND = where(scalarProducts > normProducts * (1-self.sigma))[0]
#                    if itn == 2: 
#                        print '==========='
#                        print scalarProducts
#                        print normProducts * (1-self.sigma)
#                        print '111', IND
                    if IND.size != 0:
                        #mostUseful = argmax(-asarray(usefulness)[IND])
                        #IND = delete(IND, mostUseful)
                        #indToBeRemoved +=IND.tolist()
                        indToBeRemoved += IND[:-1].tolist()
#                        if itn == 2 and IND.size == 1: 
#                            print '333', norm(objGradVectors[ind])*norm(objGradVectors[IND])/dot(objGradVectors[ind],objGradVectors[IND])
            
            indToBeRemoved = list(set(indToBeRemoved)) # TODO: simplify it
            indToBeRemoved.sort(reverse=True)

            for ind in indToBeRemoved:# TODO: simplify it
                for List in StoredInfo:
                    del List[ind]
                    
            if len(values) > nVec:
                for List in StoredInfo:
                    del List[:-nVec]
            
            #assert itn != 2
            
            print 'remains: ', len(values), '    removed: ', len(indToBeRemoved), '(',  indToBeRemoved, ')'          

#            if len(values) == 2:
#                raise 0
            
            F = asscalar(bestFeasiblePoint.f() - Ftol) if bestFeasiblePoint is not None else nan
            #F = 0.0

            #!!!!!!!! CHECK IT
            valDistances = [((values[i] - (0 if isConstraint[i] else F)) + dot(x-points[i], vec)) for i, vec in enumerate(objGradVectors)]
            #ValWRTCurrent = -1e-13 + asarray([(values[i] - (0.0 if isConstraint[i] else iterStartPoint.f()) - dot(x-points[i], vec)) for i, vec in enumerate(objGradVectors)])
            ValWRTCurrent = asarray(valDistances) - iterStartPoint.f()
           
            indActive = where(ValWRTCurrent < 0)[0]
            indInactive = where(ValWRTCurrent >= 0)[0]

            if indInactive.size != 0: 
                for k in indInactive.tolist():
                    inactive[k] += 1
            if indActive.size != 0: 
                for k in indActive.tolist():
                    inactive[k] = 0
            
            indInactiveToBeRemoved = where(asarray(inactive) > 10)[0]
            #print 'Active: ',indActive, '    inactive: ', inactive, '  indInactiveToBeRemoved:',  indInactiveToBeRemoved
            _polyedr = asarray([objGradVectors[k] for k in indActive.tolist()])
            _norms = asarray([vectorNorms[k] for k in indActive.tolist()])
            _valDistances = [valDistances[k] for k in indActive.tolist()]
            
            if indInactiveToBeRemoved.size != 0: # elseware error in current Python 2.6
                for j in indInactiveToBeRemoved:
                    for List in StoredInfo + [valDistances]:
                        del List[j]
            
            m = len(indActive)
            print 'len(indActive):', len(indActive)
            if m == 0:
                g1 = iterStartPoint._getDirection(self.approach, currBestFeasPoint = bestFeasiblePoint)
                #raise 0
            elif m == 1:
                g1 = _polyedr[0]
            elif m == 2:
                projection1 = PolytopProjection(_polyedr, asfarray(_valDistances))
                iterStartPoint = p.point(x - projection1)
                x = iterStartPoint.x            
                tmp1, tmp2 = _polyedr[0]/_norms[0] ,  _polyedr[1]/_norms[1]
                #print '>>>>>>>', dot(tmp1, tmp2)#, tmp1, tmp2
                #g1 = _polyedr[0]/_norms[0] + _polyedr[1]/_norms[1]
                g1 = _polyedr[0] + _polyedr[1]

            else:
                if self.dual:
                    projection1 = PolytopProjection(_polyedr, asfarray(_valDistances))
                    if self.ls_direction == 'simple':
#                        iterStartPoint = p.point(x - projection1)
#                        x = iterStartPoint.x
                        g1 = iterStartPoint._getDirection(self.approach, currBestFeasPoint = bestFeasiblePoint)
                    elif self.ls_direction == 'socp':
                        _polyedr_normed = asarray([objGradVectors[k]/vectorNorms[k] for k in indActive.tolist()])
                        
                        
                        from FuncDesigner import oovars, dot as DOT
                        from openopt import NLP
                        y, t = oovars(2)
                        #cons = [DOT(_polyedr_normed, y)<t, (y**2).sum()<=1]
                        cons = [DOT(_polyedr, y)<t, (y**2).sum()<=1]
                        _p = NLP(t, {t:0, y:zeros(n)}, constraints = cons, iprint = -1)
                        _r = _p.solve('scipy_slsqp')
                        g1 = _r(y)
                        #print g1
                        iterStartPoint = p.point(iterStartPoint.x - projection1)
                        x = iterStartPoint.x
                        
#                        from openopt import SOCP
#                        A = hstack((_polyedr, -ones((m, 1))))
#                        _p = SOCP([0]*n+[1], A=A, b=[0]*m, C=[array([1]*n+[0])], d=[0], q=[array([0]*(n+1))], s=[1])
#                        _r = _p.solve('cvxopt_socp', iprint = -1)
#                        g1 = _r.xf[:-1]
#                        raise 0
                    elif self.ls_direction == 'double':
                        F2 = asscalar(bestFeasiblePoint.f() - 1.5*Ftol) if bestFeasiblePoint is not None else nan
                        valDistances2 = [((values[i] - (0.0 if isConstraint[i] else F2)) + dot(x-points[i], vec)) for i, vec in enumerate(objGradVectors)]
                        projection2 = PolytopProjection(polyedr, asfarray(valDistances2))
                        point1, point2 = p.point(iterStartPoint.x - projection1), p.point(iterStartPoint.x - projection2)
                        d1 = point1._getDirection(self.approach, currBestFeasPoint = bestFeasiblePoint)
                        if dot(d1, projection1 - projection2) <= 0:
                            newStartPoint = point2
                            g1 = projection1 - projection2
                        else:
                            newStartPoint = point1
                            g1 = projection2 - projection1
                        x = newStartPoint.x
                        iterStartPoint = newStartPoint

                else:
                    p.err('only dual subproblem is turned on')
#                    from scipy.sparse import eye
#                    from openopt import QP            
#                    projection2 = QP(eye(p.n, p.n), zeros_like(x), A=polyedr, b = -valDistances).solve('cvxopt_qp', iprint = -1).xf
#                    g1 = projection2

                
            if any(g1): g1 /= p.norm(g1)
            

            """                           Forward line search                          """

            bestPointBeforeTurn = iterStartPoint
            
            hs_cumsum = 0
            hs_start = hs
            for ls in xrange(p.maxLineSearch):
                hs_mult = 1.0
                if ls > 20:
                    hs_mult = 2.0
                elif ls > 10:
                    hs_mult = 1.5
                elif ls > 2:
                    hs_mult = 1.05
                hs *= hs_mult

                x -= hs * g1
                hs_cumsum += hs

                newPoint = p.point(x) if ls == 0 else iterStartPoint.linePoint(hs_cumsum/(hs_cumsum-hs), oldPoint) #  TODO: take ls into account?
                
                if self.show_nnan: p.info('ls: %d nnan: %d' % (ls, newPoint.__nnan__()))
                
                if ls == 0:
                    oldPoint = prevIter_best_ls_point#prevIterPoint
                    oldoldPoint = oldPoint
                    
                #if not self.checkTurnByGradient:
                if newPoint.betterThan(oldPoint, altLinInEq=True, bestFeasiblePoint = bestFeasiblePoint):
                    if newPoint.isFeas(True) and (bestFeasiblePoint is None or newPoint.betterThan(bestFeasiblePoint, altLinInEq=True, bestFeasiblePoint = bestFeasiblePoint)):
                        bestFeasiblePoint = newPoint
                    if newPoint.betterThan(bestPoint, altLinInEq=True): bestPoint = newPoint
                    oldoldPoint = oldPoint
                    oldPoint, newPoint = newPoint,  None
                else:
                    bestPointBeforeTurn = oldPoint
                    if not itn % 4: 
                        for fn in ['_lin_ineq', '_lin_eq']:
                            if hasattr(newPoint, fn): delattr(newPoint, fn)
                    break

            hs /= hs_mult
            if ls == p.maxLineSearch-1:
                p.istop,  p.msg = IS_LINE_SEARCH_FAILED,  'maxLineSearch (' + str(p.maxLineSearch) + ') has been exceeded'
                return


            """                          Backward line search                          """
            maxLS = 2 if ls == 0 else 2
            maxDeltaF = p.ftol / 16.0
            maxDeltaX = p.xtol / 16.0
            if itn == 0:  
                mdx = max((hs / 128.0, 128*p.xtol )) 
                maxLS = 50
            
            ls_backward = 0
                
            if self.doBackwardSearch:
                best_ls_point,  bestPointAfterTurn, ls_backward = \
                getBestPointAfterTurn(oldoldPoint, newPoint, maxLS = maxLS, maxDeltaF = maxDeltaF, \
                                      maxDeltaX = maxDeltaX, altLinInEq = True, new_bs = True)

            # TODO: extract last point from backward search, that one is better than iterPoint
            if best_ls_point.betterThan(bestPoint, altLinInEq=True): bestPoint = best_ls_point

            if best_ls_point.isFeas(True) and (bestFeasiblePoint is None or best_ls_point.betterThan(bestFeasiblePoint, altLinInEq=True, bestFeasiblePoint = bestFeasiblePoint)):
                bestFeasiblePoint = best_ls_point

            
            #best_ls_point = bestPointAfterTurn # elseware lots of difficulties
            
            """                                 Updating hs                                 """
            step_x = p.norm(best_ls_point.x - prevIter_best_ls_point.x)
            step_f = abs(best_ls_point.f() - prevIter_best_ls_point.f())
            HS.append(hs_start)
            assert ls >= 0
            LS.append(ls)
            if itn > 3:
                mean_ls = (3*LS[-1] + 2*LS[-2]+LS[-3]) / 6.0
                j0 = 3.3
                #print 'mean_ls:', mean_ls
                #print 'ls_backward:', ls_backward
                if mean_ls > j0:
                    hs = (mean_ls - j0 + 1)**0.5 * hs_start
                else:
                    #hs = (ls/j0) ** 0.5 * hs_start
                    hs = hs_start
                    if (ls == 0 and ls_backward == -maxLS) or self.zhurb!=0:
                        shift_x = step_x / p.xtol
                        shift_f = step_f / p.ftol
#                        print 'shift_x: %e    shift_f: %e' %(shift_x, shift_f)
                        RD = log10(shift_x+1e-100)
                        if best_ls_point.isFeas(True) or prevIter_best_ls_point.isFeas(True):
                            RD = min((RD, log10(shift_f + 1e-100)))
                        #print 'RD:', RD
                        if RD > 1.0:
                            mp = (0.5, (ls/j0) ** 0.5, 1 - 0.2*RD)
                            hs *= max(mp)

            """                            Handling iterPoints                            """
               

            if itn == 0:
                p.debugmsg('hs: ' + str(hs))
                p.debugmsg('ls: ' + str(ls))
            if self.showLS: p.info('ls: ' + str(ls))
            if self.show_hs: p.info('hs: ' + str(hs))
            if self.show_nnan: p.info('nnan: ' + str(best_ls_point.__nnan__()))
            if self.showRes:
                r, fname, ind = best_ls_point.mr(True)
                p.info(fname+str(ind))

           
            """                               Call OO iterfcn                                """
            if hasattr(p, '_df'): delattr(p, '_df')
            if best_ls_point.isFeas(False) and hasattr(best_ls_point, '_df'): 
                p._df = best_ls_point.df().copy()           
                
            p.iterfcn(best_ls_point)

            """                             Check stop criteria                           """

            cond_same_point = array_equal(best_ls_point.x, prevIter_best_ls_point.x)
            if cond_same_point and not p.istop:
                p.istop = 14
                p.msg = 'X[k-1] and X[k] are same'
                p.stopdict[SMALL_DELTA_X] = True
                return
            
            s2 = 0
            if p.istop and not p.userStop:
                if p.istop not in p.stopdict: p.stopdict[p.istop] = True # it's actual for converters, TODO: fix it
                if SMALL_DF in p.stopdict:
                    if best_ls_point.isFeas(False): s2 = p.istop
                    p.stopdict.pop(SMALL_DF)
                if SMALL_DELTA_F in p.stopdict:
                    # TODO: implement it more properly
                    if best_ls_point.isFeas(False) and prevIter_best_ls_point.f() != best_ls_point.f(): s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_F)
                if SMALL_DELTA_X in p.stopdict:
                    if best_ls_point.isFeas(False) or not prevIter_best_ls_point.isFeas(False) or cond_same_point: s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_X)
#                if s2 and (any(isnan(best_ls_point.c())) or any(isnan(best_ls_point.h()))) \
#                and not p.isNaNInConstraintsAllowed\
#                and not cond_same_point:
#                    s2 = 0
                    
                if not s2 and any(p.stopdict.values()):
                    for key,  val in p.stopdict.iteritems():
                        if val == True:
                            s2 = key
                            break
                p.istop = s2
                
                for key,  val in p.stopdict.iteritems():
                    if key < 0 or key in set([FVAL_IS_ENOUGH, USER_DEMAND_STOP, BUTTON_ENOUGH_HAS_BEEN_PRESSED]):
                        #p.iterfcn(bestPoint)
                        return
            """                                If stop required                                """
            
            if p.istop:
                    #p.iterfcn(bestPoint)
                    return


            """                Some final things for gsubg main cycle                """
            prevIter_best_ls_point = best_ls_point
            #moveDirection = best_ls_point._getDirection(self.approach, currBestFeasPoint = bestFeasiblePoint)
