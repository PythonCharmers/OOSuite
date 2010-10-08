from numpy import diag, array, sqrt,  eye, ones, inf, any, copy, zeros, dot, where, all, tile, sum, nan, isfinite, float64, isnan, log10, \
max, sign, array_equal, nonzero, ix_, arctan, pi, logical_not, logical_and, atleast_2d, matrix
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
    #__optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
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
    zhurb = 20

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
        prevIter_best_ls_point = bestPoint
        best_ls_point = bestPoint
        prevIter_bestPointAfterTurn = bestPoint
        bestPointBeforeTurn = bestPoint
        
        g = bestPoint._getDirection(self.approach)
        moveDirection = g
        if not any(g) and all(isfinite(g)):
            # TODO: create ENUMs
            if bestPoint.isFeas(False):
                p.istop = 14
            else:
                p.istop = -14

            p.msg = 'move direction has all-zero coords'
            return

        HS = []
        LS = []
        
        SwitchEncountered = False
        

        """                           gsubg main cycle                                    """

        for itn in xrange(1500000):
            
            iterStartPoint = prevIter_best_ls_point
            x = iterStartPoint.x.copy()

            if itn == 0:
                objGradVectors = []
                normed_objGradVectors = []
                objValues = []
                points = []
            schedule = [bestPointBeforeTurn] #if itn == 0 else [bestPointBeforeTurn, bestPointAfterTurn]
            for point in schedule:
                if isfinite(point.f()):
                    objGradVectors.append(point.df())
                    normed_objGradVectors.append(point.df()/norm(point.df()))
                    objValues.append(asscalar(point.f()))
                    F = bestPoint.f()
                    if F == objValues[-1]: 
                        F -= p.ftol 
                    F = 0.0                    
                    points.append(point.x)
                if not point.isFeas(True):
                    objGradVectors.append(-point._getDirection(self.approach))
                    objValues.append(point.mr_alt())
                    F = 0.0
                    points.append(point.x)
                
            # simplest way to get rid of insufficient vectors
            # !!!!!!! TODO: implement it properly
            while self.zhurb < len(objValues):
                objValues.pop(0)
                objGradVectors.pop(0)
                normed_objGradVectors.pop(0)
                points.pop(0)
            

            # TODO: use matrix operations instead of the cycle
            distances = [objValues[i] - F - dot(x-points[i], vec) for i, vec in enumerate(objGradVectors)]
            polytop = []
            maxNorm = 0.0
            norms = []
                
            polytop = asarray(normed_objGradVectors)*reshape(distances, (-1, 1))
            maxNorm = sqrt(max((polytop**2).sum(1)))
            polytop /= maxNorm

            g1 = moveDirection
            if any(g1): g1 /= norm(g1)
            
            if len(objGradVectors) > 1:
                projection = PolytopProjection(polytop)
                g1 = projection
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
                if newPoint.betterThan(oldPoint, altLinInEq=True):
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
            maxLS = 300 if ls == 0 else 4
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

            
            best_ls_point = bestPointAfterTurn # elseware lots of difficulties
            
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
                        p.iterfcn(bestPoint)
                        return
            """                                If stop required                                """
            
            if p.istop:
                    p.iterfcn(bestPoint)
                    return


            """                Some final things for gsubg main cycle                """
            prevIter_best_ls_point = best_ls_point
            prevIter_best_ls_point = best_ls_point
            prevDirectionForDilation = best_ls_point._getDirection(self.approach)
            moveDirection = best_ls_point._getDirection(self.approach)
