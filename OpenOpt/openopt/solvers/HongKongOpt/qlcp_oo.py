from openopt.kernel.baseSolver import baseSolver
from QPSolve import QPSolve

class qlcp(baseSolver):
    __license__ = "MIT"
    __authors__ = "Enzo Michelangeli"
    #_requiresBestPointDetection = True
    
    __name__ = 'qlcp'
    __alg__ = 'Lemke algorithm, using linear complementarity problem'
    #__isIterPointAlwaysFeasible__ = True
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'A', 'b', 'Aeq', 'beq']
    
    def __init__(self): pass
    def __solver__(self, p):
        x,  retcode = QPSolve(p.f, p.H, p.A, p.b, p.Aeq, p.beq, p.lb, p.ub)
        
        p.istop = 1000
        # TODO: istop, msg wrt retcode
        
        p.xf = x
