from numpy import asarray,  ones, all, isfinite, copy, nan, concatenate, array, hstack, vstack
from openopt.kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from cvxopt_misc import *
import cvxopt.solvers as cvxopt_solvers
from cvxopt.base import matrix
from openopt.kernel.setDefaultIterFuncs import SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON,  IS_MAX_ITER_REACHED, IS_MAX_TIME_REACHED, FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON, UNDEFINED

def CVXOPT_SOCP_Solver(p, solverName):
    if solverName == 'native_CVXOPT_SOCP_Solver': solverName = None
    if p.iprint <= 0:
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['LPX_K_MSGLEV'] = 0
        cvxopt_solvers.options['MSK_IPAR_LOG'] = 0
    xBounds2Matrix(p)
    #FIXME: if problem is search for MAXIMUM, not MINIMUM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    f = copy(p.f).reshape(-1,1)

    # CVXOPT has some problems with x0 so currently I decided to avoid using the one
    
    Gq, hq = [], []
    C, d, q, s = p.C, p.d, p.q, p.s
    for i in xrange(len(q)):
        Gq.append(Matrix(vstack((-q[i],-C[i]))))
        hq.append(matrix(hstack((s[i], d[i])), tc='d'))

    sol = cvxopt_solvers.socp(Matrix(p.f), Gq=Gq, hq=hq, A=Matrix(p.Aeq), b=Matrix(p.beq), solver=solverName)
    p.msg = sol['status']
    if p.msg == 'optimal' :  p.istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
    else: p.istop = -100
    if sol['x'] is not None:
        p.xf = asarray(sol['x']).flatten()
        p.ff = sum(p.dotmult(p.f, p.xf))
        
    else:
        p.ff = nan
        p.xf = nan*ones([p.n,1])
