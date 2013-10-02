from openopt.kernel.baseSolver import *
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
from wh import wh

def FuncWrapper(p, attr):
    Func = getattr(p, attr)
    func = lambda x: Func(x.flatten())
    return func

class wh_conn_solver(baseSolver):
    __license__ = "proprietary"
    __authors__ = ""
    __alg__ = ""
    iterfcnConnected = True
    _canHandleScipySparse = True
    matlab = 'matlab'
    arrAttribs = ()
    funcAttribs = ()

    def __init__(self): 
        pass
    def __solver__(self, p):
        # TODO: cons patterns
#        if p.nc != 0: r.append(p._getPattern(p.user.c))
#        if p.nh != 0: r.append(p._getPattern(p.user.h))
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        
        Data = {'p': p, 'TolFun': p.ftol, \
        'TolCon': p.contol, 'TolX': p.xtol, 'solver_id': self.solver_id}
        for attr in self.arrAttribs:
            Data[attr] = getattr(p, attr)
        for attr in self.funcAttribs:
            Data[attr] = FuncWrapper(p, attr)#getattr(p, attr)(x.flatten())

        wh(Data, self.matlab)
        if p.istop == 0:
            p.istop = 1000

