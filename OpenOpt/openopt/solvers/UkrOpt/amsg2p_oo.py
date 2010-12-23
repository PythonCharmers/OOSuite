from openopt.kernel.baseSolver import *
#from openopt.kernel.Point import Point
#from openopt.kernel.setDefaultIterFuncs import *

from amsg2p import amsg2p as Solver

class amsg2p(baseSolver):
    __name__ = 'amsg2p'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Petro I. Stetsyuk, amsg2p"
    __optionalDataThatCanBeHandled__ = []
    iterfcnConnected = True
    #_canHandleScipySparse = True

    #default parameters
#    T = float64
    
    showRes = False
    show_nnan = False
    gamma = 1.0
#    approach = 'all active'

    def __init__(self): pass
    def __solver__(self, p):
        #assert self.approach == 'all active'
        if not p.isUC: p.warn('Handling of constraints is not implemented properly for the solver %s yet' % self.__name__)
        if p.fOpt is None: p.err('the selver %s requires providing optimal value fOpt')
        Solver(p.f, p.df, p.x0, p.Ftol, p.Fopt, self.gamma, p.iterfcn)
        
