from ooFun import oofun
import numpy as np
from misc import FuncDesignerException

try:
    import scipy
    scipyInstalled = True
except:
    scipyInstalled = False

def scipy_quad(_func, _a, _b, **kwargs):
    if not scipyInstalled:
        raise FuncDesignerException('to use scipy_integrate_quad you should have scipy installed, see scipy.org')
    from scipy import integrate
    if not isinstance(_a, oofun):
        a = oofun(lambda: _a, input = [], discrete=True)
    else:
        a = _a
    if not isinstance(_b, oofun):
        b = oofun(lambda: _b, input = [], discrete=True)
    else:
        b = _b


    if not isinstance(_func, oofun):
        func = _func
    else:
        func = _func.fun
        
    r = oofun(lambda x, y: integrate.quad(func, x, y, **kwargs)[0], input = [a, b])
    r.d = (lambda x, y: -func(x), lambda x, y: func(y))
    return r
    
#def 
    
