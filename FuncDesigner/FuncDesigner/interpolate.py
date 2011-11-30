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
    univariate_spline = interpolate.UnivariateSpline(*args, **kwargs)
    
    return FuncDesignerSplineGenerator(univariate_spline, *args, **kwargs)
    
        # TODO: check does isCostly = True better than False for small-scale, medium-scale, large-scale
#    return FuncDesignerSplineGenerator

class FuncDesignerSplineGenerator:
    def __call__(self, INP):
        us = self._un_sp
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
            tmp = us.__call__(x.flatten() if x.ndim > 1 else x)
            return tmp if x.ndim <= 1 else tmp.reshape(x.shape)
        r = oofun(f, INP, d = d, isCostly=True, vectorized=True)
        diffX, diffY = np.diff(self._X), np.diff(self._Y)
        
        if (all(diffX >= 0) or all(diffX <= 0)) and (all(diffY >= 0) or all(diffY <= 0)) and self._k in (1, 3):
            r.criticalPoints = False
        else:
            def _interval(*args, **kw):
                raise FuncDesignerException('''
                Currently interval calculations are implemented for 
                sorted monotone splines with order 1 or 3 only''')
            r._interval = _interval
        def Plot():
            print('Warning! Plotting spline is recommended from FD spline generator, not initialized spline')
            self.plot()
        def Residual():
            print('Warning! Getting spline residual is recommended from FD spline generator, not initialized spline')
            return self.residual()
            
        r.plot, r.residual = Plot, Residual
        return r
        
    def __init__(self, us, *args, **kwargs):
        self._un_sp = us
        self._X, self._Y = args[0], args[1]
        
        if len(args) >= 5:
            k = args[4]
        elif 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 3 # default for UnivariateSpline
            
        self._k = k
    
    def plot(self):
        try:
            import pylab
        except:
            print('You should have matplotlib installed')
            return
        pylab.scatter(self._X, self._Y, marker='o')
       
        YY = self._un_sp.__call__(self._X)
        pylab.plot(self._X, YY)
        
        pylab.grid('on')
        pylab.title('FuncDesigner spline checker')
        pylab.show()
    
    def residual(self):
        YY = self._un_sp.__call__(self._X)
        return np.max(np.abs(YY - self._Y))

