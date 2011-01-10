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
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars', 'H']
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
            senses = where(p.dwhole == -1, 'L', 'E')
            P.linear_constraints.add(rhs=np.asarray(p.bwhole).tolist(),  senses = senses)
            rows,  cols,  vals = Find(p.Awhole)
            P.linear_constraints.set_coefficients(zip(rows, cols, vals))
        
        if p.probType.endswith('QP') or p.probType == 'SOCP':
            assert p.probType in ('QP', 'QCQP','SOCP')
            rows,  cols,  vals = Find(p.H)
            P.objective.set_quadratic_coefficients(zip(rows,  cols,  vals))
            
            #P.quadratic_constraints.add()
            #raise 0

        P.solve()
        
#        class StandardOutputEater:
#            def write(self, string):
#                pass
#        S, sys.stderr = sys.stderr, StandardOutputEater
#        try:
#            P.solve()
#        finally:
#            #pass
#            sys.stderr = S
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
    if type(M) == np.ndarray:
        rows, cols = np.where(M)
        vals = M[rows,cols]
    else:
        from scipy import sparse as sp
        assert sp.isspmatrix(M)
        rows, cols, vals = sp.find(M)
    return rows.tolist(), cols.tolist(), vals.tolist()
