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
        if p.fOpt is None: p.err('the solver %s requires providing optimal value fOpt')
        if p.Ftol is None: 
            s = '''
            the solver %s requires providing required objective function tolerance Ftol
            15*ftol = %0.1e will be used instead
            ''' % (self.__name__, p.ftol)
            p.pWarn(s)
            Ftol = 15*p.ftol
        else: Ftol = p.Ftol
        x, itn = Solver(p.f, p.df, p.x0, Ftol, p.fOpt, self.gamma, p.iterfcn)
        if p.f(x) < p.fOpt + Ftol:
            p.istop = 10
        #p.iterfcn(x)
