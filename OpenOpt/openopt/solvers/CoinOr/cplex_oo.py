import sys, numpy as np
from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.setDefaultIterFuncs import *
from openopt.kernel.ooMisc import LinConst2WholeRepr
#from openopt.kernel.ooMisc import isSolved
#from openopt.kernel.nonOptMisc import scipyInstalled, Hstack, Vstack, Find, isspmatrix
import os
    
class cplex(baseSolver):
    __name__ = 'cplex'
    __license__ = "free for academic"
    #__authors__ = ''
    #__alg__ = ""
    #__homepage__ = 'http://www.coin-or.org/'
    #__info__ = ""
    #__cannotHandleExceptions__ = True
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars', 'H', 'QC']
    _canHandleScipySparse = True
    
    #options = ''

    def __init__(self): pass
    def __solver__(self, p):
        try:
            import cplex
        except ImportError:
            p.err('You should have Cplex and its Python API installed')
        
        n = p.f.size
        
        # reduce text output
        os.close(1); os.close(2) # may not work for non-Unix OS
         
        P = cplex.Cplex()
        P.set_results_stream(None)
        
        if np.isfinite(p.maxTime): 
            P.parameters.timelimit.set(p.maxTime)
        
        kwargs = {}
        if hasattr(p, 'intVars') and len(p.intVars)!=0: 
            tmp = np.asarray(['C']*n, dtype=object)
            for v in p.intVars: tmp[v] = 'I'
            kwargs['types'] = ''.join(tmp.tolist())
        
        P.variables.add(obj = p.f.tolist(), ub = p.ub.tolist(), lb = p.lb.tolist(), **kwargs)
        P.objective.set_sense(P.objective.sense.minimize)
        
        LinConst2WholeRepr(p)
        if p.Awhole is not None:
            m = np.asarray(p.bwhole).size
            senses = ''.join(where(p.dwhole == -1, 'L', 'E').tolist())
            P.linear_constraints.add(rhs=np.asarray(p.bwhole).tolist(), senses = senses)
            P.linear_constraints.set_coefficients(zip(*Find(p.Awhole)))
        
        if p.probType.endswith('QP') or p.probType == 'SOCP':
            assert p.probType in ('QP', 'QCQP','SOCP')
            P.objective.set_quadratic_coefficients(zip(*Find(p.H)))
            if hasattr(p, 'QC'):
                for q, u, v in p.QC:
                    rows,  cols,  vals = Find(q)
                    quad_expr = cplex.SparseTriple(ind1=rows, ind2=cols, val = vals)
                    #lin_expr = zip(np.arange(np.atleast_1d(u).size), u)
                    lin_expr = cplex.SparsePair(ind=np.arange(np.atleast_1d(u).size), val=u)
                    P.quadratic_constraints.add(quad_expr = quad_expr, lin_expr = lin_expr, rhs = -v if isscalar(v) else -asscalar(v))

            #P.quadratic_constraints.add()
            #raise 0
        #raise 0
        
        # Temporary walkaround Cplex 12.2.0.0 bug with integers in QP/QCQP
        P.SOS.get_num()
        
        P.solve()
        s = P.solution.get_status()
        p.msg = 'Cplex status: "%s"; exit code: %d' % (P.solution.get_status_string(), s)
        try:
            p.xf = np.asfarray(P.solution.get_values())
            p.istop = 1000
        except cplex.exceptions.CplexError:
            p.xf = p.x0 * np.nan
            p.istop = -1
        
        # TODO: replace by normal OOP solution
        if s == P.solution.status.abort_iteration_limit:
            p.istop = IS_MAX_ITER_REACHED
            p.msg = 'Max Iter has been reached'
        elif s == P.solution.status.abort_obj_limit:
            p.istop = IS_MAX_FUN_EVALS_REACHED
            p.msg = 'max objfunc evals limit has been reached'
        elif s == P.solution.status.abort_time_limit or s == P.solution.status.conflict_abort_time_limit:
            p.istop = IS_MAX_TIME_REACHED
            p.msg = 'max time limit has been reached'
            

def Find(M):
    if isinstance(M, np.ndarray): # numpy array or matrix
        rows, cols = np.where(M)
        vals = M[rows,cols]
    else:
        from scipy import sparse as sp
        assert sp.isspmatrix(M)
        rows, cols, vals = sp.find(M)
    return rows.tolist(), cols.tolist(), vals.tolist()
