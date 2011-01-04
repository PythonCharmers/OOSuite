import numpy as np
from openopt.kernel.baseSolver import baseSolver
#from openopt.kernel.ooMisc import isSolved
#from openopt.kernel.nonOptMisc import scipyInstalled, Hstack, Vstack, Find, isspmatrix

    
class cplex(baseSolver):
    __name__ = 'cplex'
    __license__ = "free for academic"
    #__authors__ = ''
    #__alg__ = ""
    #__homepage__ = 'http://www.coin-or.org/'
    #__info__ = ""
    #__cannotHandleExceptions__ = True
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    _canHandleScipySparse = True

    #options = ''

    def __init__(self): pass
    def __solver__(self, p):
        try:
            import cplex
        except ImportError:
            p.err('You should have Cplex and its Python API installed')
        
        P = cplex.Cplex()
        P.variables.add(obj = p.f.tolist(), ub = p.ub.tolist(), lb = p.lb.tolist())
        P.objective.set_sense(P.objective.sense.minimize)
        n = p.f.size
        
        for _A, _b, _T in [(p.A, p.b,'L'), (p.Aeq, p.beq, 'E')]:
            if _b is None or np.asarray(_b).size == 0:
                continue
            m = np.asarray(_b).size
            P.linear_constraints.add(rhs = np.asarray(_b).tolist(), senses = _T*m)
            if type(_A) == np.ndarray:
                rows = np.tile(np.arange(m).reshape(-1, 1), (1, n)).flatten()
                cols = np.asarray([np.arange(n).tolist()]*m).flatten()
                vals = _A.flatten()
            else:
                from scipy import sparse as sp
                assert sp.isspmatrix(_A)
                rows, cols, vals = sp.find(_A)
        
        P.linear_constraints.set_coefficients(zip(rows.tolist(), cols.tolist(), vals.tolist()))
        P.solve()
        p.xf = np.asfarray(P.solution.get_values())
