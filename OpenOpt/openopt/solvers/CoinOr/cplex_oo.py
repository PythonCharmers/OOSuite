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
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars']
    _canHandleScipySparse = True

    #options = ''

    def __init__(self): pass
    def __solver__(self, p):
        try:
            import cplex
        except ImportError:
            p.err('You should have Cplex and its Python API installed')
        
        n = p.f.size
        
        P = cplex.Cplex()
        kwargs = {}
        if hasattr(p, 'intVars') and len(p.intVars)!=0: 
            tmp = np.asarray(['C']*n, dtype=object)
            for v in p.intVars: tmp[v] = 'I'
            kwargs['types'] = ''.join(tmp.tolist())
        
        P.variables.add(obj = p.f.tolist(), ub = p.ub.tolist(), lb = p.lb.tolist(), **kwargs)
        P.objective.set_sense(P.objective.sense.minimize)
        
        for _A, _b, _T in [(p.A, p.b,'L'), (p.Aeq, p.beq, 'E')]:
            if _b is None or np.asarray(_b).size == 0:
                continue
            m = np.asarray(_b).size
            P.linear_constraints.add(rhs=np.asarray(_b).tolist(),  senses= _T*m)
            rows,  cols,  vals = Find(_A)
            P.linear_constraints.set_coefficients(zip(rows, cols, vals))
        
        if p.probType.endswith('QP'):
            assert p.probType in ('QP', 'QCQP')
            #p.objective.set_quadratic_coefficients()
            
        P.solve()
        p.xf = np.asfarray(P.solution.get_values())

def Find(M):
    if type(M) == np.ndarray:
        rows, cols = np.where(M)
        vals = M[rows,cols]
    else:
        from scipy import sparse as sp
        assert sp.isspmatrix(M)
        rows, cols, vals = sp.find(M)
    return rows.tolist(), cols.tolist(), vals.tolist()
