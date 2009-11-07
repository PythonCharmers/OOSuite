from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d,  zeros, ones, int, float64, where, inf, linalg, ndarray
from openopt.kernel.baseSolver import baseSolver

try:
    import scipy
except:
    pass

class defaultSLEsolver(baseSolver):
    __name__ = 'defaultSLEsolver'
    __license__ = "BSD"
    __authors__ = 'Dmitrey'
    __alg__ = ''
    __info__ = ''
    #__optionalDataThatCanBeHandled__ = []

    def __init__(self): pass

    def __solver__(self, p):
        if isinstance(p.C, ndarray):
            try:
                xf = linalg.solve(p.C, p.d)
                istop, msg = 10, 'solved'
                p.xf = xf
            except linalg.LinAlgError:
                istop, msg = -10, 'singular matrix'
        else: # is sparse
            try:
                xf = scipy.sparse.linalg.spsolve(p.C.tocsc(), p.d)
                istop, msg = 10, 'solved'
                p.xf = xf                
            except:
                istop, msg = -100, 'unimplemented exception while solving sparse SLE'

        p.istop, p.msg = istop, msg

