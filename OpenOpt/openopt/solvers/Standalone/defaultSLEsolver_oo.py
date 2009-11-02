from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d,  zeros, ones, int, float64, where, inf, linalg
from openopt.kernel.baseSolver import baseSolver

class defaultSLEsolver(baseSolver):
    __name__ = 'defaultSLEsolver'
    __license__ = "BSD"
    __authors__ = 'Dmitrey'
    __alg__ = ''
    __info__ = ''
    #__optionalDataThatCanBeHandled__ = []

    def __init__(self): pass

    def __solver__(self, p):
        try:
            xf = linalg.solve(p.C, p.d)
            istop, msg = 10, 'solved'
            p.xf = xf
        except linalg.LinAlgError:
            istop, msg = -10, 'singular matrix'

        p.istop, p.msg = istop, msg

        #p.ff = p.fk = w[0]

