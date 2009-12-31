from numpy import diag, array, sqrt,  eye, ones, inf, any, copy, zeros, dot, where, all, tile, sum, nan, isfinite, float64, isnan, log10, \
max, sign, array_equal, nonzero, ix_, arctan, pi
from numpy.linalg import norm, solve, LinAlgError
#try:
#    from numpy.linalg import cond
#except:
#    print 'warning: no cond in numpy.linalg, matrix B rejuvenation check will be omitted'
#    cond = lambda Matrix: 1

from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point
from openopt.kernel.ooMisc import economyMult, Len
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F,  SMALL_DF,  IS_LINE_SEARCH_FAILED
from UkrOptMisc import getBestPointAfterTurn

class ralg(baseSolver):
    __name__ = 'ralg'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Naum Z. Shor R-algorithm with adaptive space dilation & some modifications"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    __iterfcnConnected__ = True
    _canHandleScipySparse = True

    #ralg default parameters
    alp, h0, nh, q1, q2  = 2.0, 1.0, 3, 'default:0.9 for NLP, 1.0 for NSP', 1.1
    hmult = 0.5
    S = 0
    T = float64
    dilationType = 'auto'

    showLS = False
    show_hs = False
    showRej = False
    showRes = False
    show_nnan = False
    doBackwardSearch = 1
    approach = 'nqp'
    newLinEq = True

    def needRej(self, p, b, g, g_dilated):
        return 1e15 * p.norm(g_dilated) < p.norm(g)
    #checkTurnByGradient = True

    def __init__(self): pass
    def __solver__(self, p):

        alp, h0, nh, q1, q2 = self.alp, self.h0, self.nh, self.q1, self.q2

        if type(q1) == str:
            if p.probType== 'NLP' and p.isUC: q1 = 0.9
            else: q1 = 1.0
        T = self.T
        # alternatively instead of alp=self.alp etc you can use directly self.alp etc

        n = p.n
        x0 = p.x0
        x0[x0<p.lb] = p.lb[x0<p.lb]
        x0[x0>p.ub] = p.ub[x0>p.ub]
        
        ind_box_eq = where(p.lb==p.ub)[0]
        nEQ = ind_box_eq.size
        if nEQ != 0:
            initLenBeq = p.nbeq
            Aeq, beq, nbeq = copy(p.Aeq), copy(p.beq), p.nbeq
            p.Aeq = zeros([Len(p.beq) + nEQ, p.n])
            p.beq = zeros(Len(p.beq) + nEQ)
            p.beq[:Len(beq)] = beq
            p.Aeq[:Len(beq)] = Aeq
            for i in xrange(len(ind_box_eq)):
                p.Aeq[initLenBeq+i, ind_box_eq[i]] = 1
                p.beq[initLenBeq+i] = p.lb[ind_box_eq[i]] # = p.ub[indEQ[i]], because they are the same
            p.nbeq += nEQ
            
        if not self.newLinEq or p.nbeq == 0:
            needProjection = False
            B0 = eye(n,  dtype=T)
            restoreProb = lambda *args: 0
            Aeq_r, beq_r, nbeq_r = None, None, 0
        else:
            needProjection = True
            B0 = self.getPrimevalDilationMatrixWRTlinEqConstraints(p)
            #Aeq, beq, nbeq = p.Aeq, p.beq, p.nbeq
            
            if any(abs(p.__get_AeqX_eq_Beq_residuals__(x0))>p.contol/16.0):
                #p.debugmsg('old point Aeq residual:'+str(norm(dot(Aeq, x0)-beq)))
                try:
                    x0 = self.linEqProjection(x0, p.Aeq, p.beq)
                except LinAlgError:
                    s = 'Failed to obtain projection of start point to linear equality constraints subspace, probably the system is infeasible'
                    p.istop, p.msg = -25,  s
                    return
                    
                #p.debugmsg('new point Aeq residual:'+str(norm(dot(Aeq, x0)-beq)))
            if nEQ == 0:
                Aeq_r, beq_r, nbeq_r = p.Aeq, p.beq, p.nbeq
            else:
                Aeq_r, beq_r, nbeq_r = Aeq, beq, nbeq
            
            p.Aeq, p.beq, p.nbeq = None, None, 0
            
            # TODO: return prob with unmodified Aeq, beq
            
            def restoreProb():
                p.Aeq, p.beq, p.nbeq = Aeq_r, beq_r, nbeq_r
                #if nEQ != 0: restore lb, ub
                    
            
        b = B0.copy()
#        B_f = diag(ones(n))
#        B_constr = diag(ones(n))
        hs = T(h0)
        ls_arr = []
        w = T(1.0/alp-1.0)

        """                            Shor r-alg engine                           """
        prevIterPoint = p.point(atleast_1d(T(copy(x0))))
        bestPoint = prevIterPoint

        g = prevIterPoint.__getDirection__(self.approach)
        moveDirection = g
        if not any(g) and all(isfinite(g)):
            # TODO: create ENUMs
            if prevIterPoint.isFeas():
                p.istop = 14
            else:
                p.istop = -14

            p.msg = 'move direction has all-zero coords'
            return

        #p.hs = [hs]
        
        directionVectorsList = []
#        #pass-by-ref! not copy!
#        if p.isFeas(p.x0): b = B_f
#        else: b = B_constr

        """                           Ralg main cycle                                    """

        for itn in xrange(1500000):
            doDilation = True
            alp_addition = 0.0

            g_tmp = economyMult(b.T, moveDirection)
            if any(g_tmp): g_tmp /= p.norm(g_tmp)
            g1 = p.matmult(b, g_tmp)


            """                           Forward line search                          """

            x = prevIterPoint.x.copy()
            hs_cumsum = 0
            for ls in xrange(p.maxLineSearch):
                if ls > 20:
                    hs *= 2.0
                elif ls > 10:
                    hs *= 1.5
                elif ls > 2:
                    hs *= 1.05

                x -= hs * g1
                hs_cumsum += hs

                newPoint = p.point(x)
              
                if self.show_nnan: p.info('ls: %d nnan: %d' % (ls, newPoint.__nnan__()))

                if ls == 0:
                    oldPoint = prevIterPoint
                elif ls >= 2:
                    # TODO: handle it outside of the file
                    newPoint._lin_ineq = prevIterPoint.lin_ineq() + hs_cumsum / (hs_cumsum - hs) * (oldPoint.lin_ineq() - prevIterPoint.lin_ineq())
                    # the _lin_eq is obsolete and may be ignored, provided newLinEq = True
                    newPoint._lin_eq = prevIterPoint.lin_eq() + hs_cumsum / (hs_cumsum - hs) * (oldPoint.lin_eq() - prevIterPoint.lin_eq())


                #if not self.checkTurnByGradient:
                if newPoint.betterThan(oldPoint, altLinInEq=True):
                    if newPoint.betterThan(bestPoint): bestPoint = newPoint
                    oldPoint, newPoint = newPoint,  None
                else:
                    break

            if ls == p.maxLineSearch-1:
                p.istop,  p.msg = IS_LINE_SEARCH_FAILED,  'maxLineSearch (' + str(p.maxLineSearch) + ') has been exceeded'
                restoreProb()
                return

            g2 = newPoint.__getDirection__(self.approach) # used for dilation direction obtaining

            iterPoint  = newPoint

            """                          Backward line search                          """

            ls_backward = 0
            if ls == 0:
                if self.doBackwardSearch:
                    iterPoint, ls_backward = getBestPointAfterTurn(prevIterPoint, iterPoint, maxLS = 3, altLinInEq = True)

                    # TODO: extract last point from backward search, that one is better than iterPoint
                    if iterPoint.betterThan(bestPoint): bestPoint = iterPoint
                    #p.debugmsg('ls_backward:%d' % ls_backward)
                    hs *= 2 ** ls_backward
                    
                    if ls_backward <= -2 and itn != 0:  # TODO: mb use -1 or 0 instead?
                        #pass
                        alp_addition -= 0.5*ls_backward # ls_backward less than zero
                    
                    #hs *= 2 ** min((ls_backward+1, 0))
                else:
                    pass
                    #hs *= 0.95

            """                      iterPoint has been obtained                     """

            moveDirection = iterPoint.__getDirection__(self.approach)
            # DEBUG!
            #g2 = moveDirection#newPoint.__getDirection__(self.approach)
            # DEBUG end!

#            ls_arr.append(ls)
#            if ls >= 2 and len(ls_arr) >= len(self.j_multiplier):
#                j_arr = array(ls_arr[-3:]) * self.j_multiplier#array((ls_arr[-3], 2.0*ls_arr[-2],  3.0*ls_arr[-1]))
#                j_arr[j_arr<0] = 0 # for more safety
#                j_mean = j_arr.sum() / sum(self.j_multiplier)
#                if j_mean > self.ls_0:
#                    hs *= sqrt(j_mean - self.ls_0 + 1.0)
#                else:
#                    hs *= sqrt(j_mean / self.ls_0)

            if itn == 0:
                p.debugmsg('hs: ' + str(hs))
                p.debugmsg('ls: ' + str(ls))
            if self.showLS: p.info('ls: ' + str(ls))
            if self.show_hs: p.info('hs: ' + str(hs))
            if self.show_nnan: p.info('nnan: ' + str(iterPoint.__nnan__()))
            if self.showRes:
                r, fname, ind = iterPoint.mr(True)
                p.info(fname+str(ind))

            """                         Set dilation direction                            """

            #if sum(p.dotmult(g, g2))>0:
                #p.debugmsg('ralg warning: slope angle less than pi/2. Mb dilation for the iter will be omitted.')
                #doDilation = False

            prevIterPointIsFeasible = prevIterPoint.isFeas(altLinInEq=True)
            currIterPointIsFeasible = iterPoint.isFeas(altLinInEq=True)
            r_p, ind_p, fname_p = prevIterPoint.mr(1)
            r_, ind_, fname_ = iterPoint.mr(1)

            if self.dilationType == 'normalized' and (not fname_p in ('lb', 'ub', 'lin_eq', 'lin_ineq') or not fname_ in ('lb', 'ub', 'lin_eq', 'lin_ineq')) and (fname_p != fname_  or ind_p != ind_):
                G2,  G = g2/norm(g2), g/norm(g)
            else:
                G2,  G = g2, g

            # CHANGES
            gn = g2/norm(g2)
            if len(directionVectorsList) == 0 or n < 3: pass
            else:
                if len(directionVectorsList) == 1 or abs(dot(directionVectorsList[-1], directionVectorsList[-2]))>0.999:
                    projectionComponentLenght = abs(dot(directionVectorsList[-1], gn))
                    restLength = sqrt(1 - min((1, projectionComponentLenght))**2)
                else: 
                    e1 = directionVectorsList[-1]
                    e2 = directionVectorsList[-2] - dot(directionVectorsList[-1], directionVectorsList[-2]) * directionVectorsList[-1]
                    e2 /= norm(e2)
                    proj1, proj2 = dot(directionVectorsList[-1], gn), dot(directionVectorsList[-2], gn)
                    rest = gn - proj1 * directionVectorsList[-1] - proj2 * directionVectorsList[-2]
                    restLength = norm(rest)
                assert restLength < 1+1e-5, 'error in ralg solver: incorrect restLength'
                
                # TODO: make it parameters of ralg
                commonCoeff, alp_add_coeff = 0.5, 1.0
                
                if restLength < commonCoeff * (n - 2.0) / n:
                    #pass
                    alpAddition = 0.5+(arctan((n - 2.0) / (n * restLength)) - pi / 4.0) / (pi / 2.0) * alp_add_coeff
                    #p.debugmsg('alpAddition:' + str(alpAddition))
                    assert alpAddition > 0 # if someone incorrectly modifies commonCoeff it can be less than zero
                    alp_addition += alpAddition
                    #p.debugmsg('alp_addition:' + str(alp_addition))
                    
            directionVectorsList.append(gn)
            if len(directionVectorsList) > 2: directionVectorsList = directionVectorsList[:-2]
            # CHANGES END

            if prevIterPointIsFeasible == currIterPointIsFeasible == True:
                g1 = G2 - G
            elif prevIterPointIsFeasible == currIterPointIsFeasible == False:
                g1 = G2 - G
            elif prevIterPointIsFeasible:
                g1 = G2.copy()
            else:
                g1 = G.copy()
                #g1 = -G.copy() # signum doesn't matter here


            """                             Perform dilation                               """

            # DEBUG!
#            W = w
#            W = T(1.0/alp-1.0) if prevIterPointIsFeasible == currIterPointIsFeasible else T(1.0/(16*alp)-1.0)
#            W = T(1.0/(2*alp)-1.0)
            # DEBUG END

            g = economyMult(b.T, g1)
            ng = p.norm(g)
            p._df = g2.copy()
            #if p.iter>500: p.debugmsg(str(g2))

            if self.needRej(p, b, g1, g):
                if self.showRej or p.debug:
                    p.info('debug msg: matrix B restoration in ralg solver')
                b = B0.copy()
                hs = 0.5*p.norm(prevIterPoint.x - iterPoint.x)
            #p.debugmsg('ng:%e  ng1:%e' % (ng, p.norm(g1)))
            if ng < 1e-40: 
                #raise 0
                hs *= 0.9
                p.debugmsg('small dilation direction norm (%e), skipping' % ng)
            #p.debugmsg('dilation direction norm:%e' % ng)
            if all(isfinite(g)) and ng > 1e-50 and doDilation:
                g = (g / ng).reshape(-1,1)
                vec1 = economyMult(b, g).reshape(-1,1)# TODO: remove economyMult, use dot?
                #if alp_addition != 0: p.debugmsg('alp_addition:' + str(alp_addition))
                w = T(1.0/(alp+alp_addition)-1.0) 
                vec2 = w * g.T
                b += p.matmult(vec1, vec2)
            
            # DEBUG!!
            #print '>!', p.matmult(iterPoint.x, g1)
            
#            if itn == 0: COS = []
#            from numpy import arange, diff
#            P = arange(n) # point
#            direction = -arange(n)
#            subgrad = -arange(n)* (1+arange(n)/(n+0.1))
#            direction, subgrad = direction/norm(direction), subgrad/norm(subgrad)
#            
#            subgrad_dilated = economyMult(b.T, subgrad)
#            if any(subgrad_dilated): subgrad_dilated /= p.norm(subgrad_dilated)
#            subgrad_dilated = p.matmult(b, subgrad_dilated)
#            cos_a = p.matmult(subgrad_dilated, direction)
#            COS.append(cos_a)
#            if itn == 90: print '>>', COS, diff(COS)
            #stat_vec = range(n)
            
            
            # DEBUG END

            """                               Call OO iterfcn                                """
            p.iterfcn(iterPoint)


            """                             Check stop criteria                           """

            cond_same_point = array_equal(iterPoint.x, prevIterPoint.x)
            if cond_same_point and not p.istop:
                p.istop = 14
                p.msg = 'X[k-1] and X[k] are same'
                p.stopdict[SMALL_DELTA_X] = True
                restoreProb()
                return
            
            s2 = 0
            if p.istop and not p.userStop:
                if p.istop not in p.stopdict: p.stopdict[p.istop] = True # it's actual for converters, TODO: fix it
                if SMALL_DF in p.stopdict:
                    if currIterPointIsFeasible: s2 = p.istop
                    p.stopdict.pop(SMALL_DF)
                if SMALL_DELTA_F in p.stopdict:
                    # TODO: implement it more properly
                    if currIterPointIsFeasible and prevIterPoint.f() != iterPoint.f(): s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_F)
                if SMALL_DELTA_X in p.stopdict:
                    if currIterPointIsFeasible or not prevIterPointIsFeasible or cond_same_point: s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_X)
                if s2 and (any(isnan(iterPoint.c())) or any(isnan(iterPoint.h()))) \
                and not p.isNaNInConstraintsAllowed\
                and not cond_same_point:
                    s2 = 0
                    
                if not s2 and any(p.stopdict.values()):
                    for key,  val in p.stopdict.iteritems():
                        if val == True:
                            s2 = key
                            break
                p.istop = s2


            """                                If stop required                                """
            
            if p.istop:
                if self.needRej(p, b, g1, g):
                    b = B0.copy()
                    hs = 0.5*p.norm(prevIterPoint.x - iterPoint.x)
                    p.istop = 0
                else:
                    restoreProb()
                    p.iterfcn(bestPoint)
                    #p.istop, p.msg = istop, msg
                    return


            """                Some final things for ralg main cycle                """
#            p.debugmsg('new point Aeq residual:'+str(norm(dot(Aeq, iterPoint.x)-beq)))
#            if needProjection and itn!=0:
#                #pass
#                x2 = self.linEqProjection(iterPoint.x, Aeq, beq)
#                p.debugmsg('norm(delta):' + str(norm(iterPoint.x-x2))) 
#                iterPoint = p.point(x2)
#                p.debugmsg('2: new point Aeq residual:'+str(norm(dot(Aeq, iterPoint.x)-beq)))
            #p.hs.append(hs)
            #g = moveDirection.copy()
            g = g2.copy()

            prevIterPoint, iterPoint = iterPoint, None


    def getPrimevalDilationMatrixWRTlinEqConstraints(self, p):
        n, Aeq, beq = p.n, p.Aeq, p.beq
        nLinEq = len(p.beq)
        ind_fixed = where(p.lb==p.ub)[0]
        arr=ones(n, dtype=self.T)
        arr[ind_fixed] = 0
        b = diag(arr)

        for i in xrange(nLinEq):
            g = Aeq[i]
            g = p.matmult(b.T, g)
            #ind_nnz = nonzero(g)[0]
            ng = norm(g)
            g = (g / ng).reshape(-1,1)
            
            vec1 = economyMult(b, g)# TODO: remove economyMult, use dot?
            vec2 = -g.T
            
            b += p.matmult(vec1, vec2)
            
#            if len(ind_nnz) > 0.7 * g.size:
#                b += p.matmult(vec1, vec2)
#            else:
#                ind_nnz1 = nonzero(vec1)[0]
#                ind_nnz2 = nonzero(vec2)[1]
#                r = dot(vec1[ind_nnz1, :], vec2[:, ind_nnz2])
#                if p.debug: 
#                    assert abs(norm(p.matmult(vec1, vec2).flatten()) - norm(r.flatten())) < 1e-5
#                b[ix_(ind_nnz1, ind_nnz2)] += r
 
        return b
 
    def linEqProjection(self, x, Aeq, beq):
        # TODO: handle case nbeq = 1 ?
        AeqT = Aeq.T
        AeqAeqT = dot(Aeq, AeqT)
        Aeqx = dot(Aeq, x)
        AeqT_AeqAeqT_inv_Aeqx = dot(AeqT, ravel(solve(AeqAeqT, Aeqx)))
        AeqT_AeqAeqT_inv_beq = dot(AeqT, ravel(solve(AeqAeqT, beq)))
        xf= x - AeqT_AeqAeqT_inv_Aeqx + AeqT_AeqAeqT_inv_beq
        return xf
        
        
        
