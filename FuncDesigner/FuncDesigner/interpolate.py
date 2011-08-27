from ooFun import oofun
import numpy as np
from FDmisc import FuncDesignerException

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
            #if x.size != 1:
                #raise FuncDesignerException('for scipy_UnivariateSpline input should be oovar/oofun with output size = 1,other cases not implemented yet')
            return us.__call__(x, 1)
        def f(x):
            x = np.asfarray(x)
            #if x.size != 1:
                #raise FuncDesignerException('for scipy_UnivariateSpline input should be oovar/oofun with output size = 1,other cases not implemented yet')            
            return us.__call__(x)
        r = oofun(f, INP, d = d, isCostly=True, vectorized=True)
        diffX, diffY = np.diff(args[0]), np.diff(args[1])
        if len(args) >= 5:
            k = args[4]
        elif 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 3 # default for UnivariateSpline
        if (all(diffX >= 0) or all(diffX <= 0)) and (all(diffY >= 0) or all(diffY <= 0)) and k in (1, 3):
            r.criticalPoints = False
        else:
            def _interval(*args, **kw):
                raise FuncDesignerException('''
                Currently interval calculations are implemented for 
                sorted monotone splines with order 1 or 3 only''')
            r._interval = _interval
        return r
        # TODO: check does isCostly = True better than False for small-scale, medium-scale, large-scale
    return makeOutput

