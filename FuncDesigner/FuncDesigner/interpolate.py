from ooFun import oofun
import numpy as np
from misc import FuncDesignerException

try:
    from scipy import interpolate
    scipyInstalled = True
except:
    scipyInstalled = False

def scipy_UnivariateSpline(*args, **kwargs):
    if not scipyInstalled:
        raise FuncDesignerException('to use scipy_UnivariateSpline you should have scipy installed, see scipy.org')
    assert len(args)>1 
    assert not isinstance(args[0], oofun) and not isinstance(args[1], oofun), \
    'init scipy splines from oovar/oofun content is not implemented yet'
    us = interpolate.UnivariateSpline(*args, **kwargs)
    def makeOutput(INP):
        if not isinstance(INP, oofun):
            raise FuncDesignerException('for scipy_UnivariateSpline input should be oovar/oofun,other cases not implemented yet')
        def d(x):
            x = np.asfarray(x)
            if x.size != 1:
                raise FuncDesignerException('for scipy_UnivariateSpline input should be oovar/oofun with output size = 1,other cases not implemented yet')
            return us.__call__(x, 1)
        def f(x):
            x = np.asfarray(x)
            if x.size != 1:
                raise FuncDesignerException('for scipy_UnivariateSpline input should be oovar/oofun with output size = 1,other cases not implemented yet')            
            return us.__call__(x)
        return oofun(f, INP, d = d, isCostly=True)
        # TODO: check does isCostly = True better than False for small-scale, medium-scale, large-scale
    return makeOutput

