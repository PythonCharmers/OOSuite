from openopt.kernel.baseSolver import baseSolver
from CVXOPT_LP_Solver import CVXOPT_LP_Solver

class glpk(baseSolver):
    __name__ = 'glpk'
    __license__ = "GPL v.2"
    __authors__ = "http://www.gnu.org/software/glpk + Python bindings from http://abel.ee.ucla.edu/cvxopt"
    __homepage__ = 'http://www.gnu.org/software/glpk'
    #__alg__ = ""
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars', 'binVars']

    def __init__(self): pass

    def __solver__(self, p):
        return CVXOPT_LP_Solver(p, 'glpk')
