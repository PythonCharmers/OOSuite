from ooFun import oofun
import numpy as np
from FDmisc import FuncDesignerException

try:
    from scipy import interpolate
    scipyInstalled = True
except:
    scipyInstalled = False

def scipy_InterpolatedUnivariateSpline(*args, **kwargs):
    if not scipyInstalled:
        raise FuncDesignerException('to use scipy_InterpolatedUnivariateSpline you should have scipy installed, see scipy.org')
    assert len(args)>1 
    assert not isinstance(args[0], oofun) and not isinstance(args[1], oofun), \
    'init scipy splines from oovar/oofun content is not implemented yet'
    S = interpolate.InterpolatedUnivariateSpline(*args, **kwargs)
    
    return SplineGenerator(S, *args, **kwargs)
    
        # TODO: check does isCostly = True better than False for small-scale, medium-scale, large-scale
#    return SplineGenerator

class SplineGenerator:
    def __call__(self, INP):
        us = self._un_sp
        if not isinstance(INP, oofun):
            raise FuncDesignerException('for scipy_InterpolatedUnivariateSpline input should be oovar/oofun,other cases not implemented yet')
        def d(x):
            x = np.asfarray(x)
            #if x.size != 1:
                #raise FuncDesignerException('for scipy_InterpolatedUnivariateSpline input should be oovar/oofun with output size = 1,other cases not implemented yet')
            return us.__call__(x, 1)
        def f(x):
            x = np.asfarray(x)
            #if x.size != 1:
                #raise FuncDesignerException('for scipy_InterpolatedUnivariateSpline input should be oovar/oofun with output size = 1,other cases not implemented yet')            
            tmp = us.__call__(x.flatten() if x.ndim > 1 else x)
            return tmp if x.ndim <= 1 else tmp.reshape(x.shape)
        r = oofun(f, INP, d = d, isCostly=True, vectorized=True)
        r._nonmonotone_x = self._nonmonotone_x
        r._nonmonotone_y = self._nonmonotone_y
        
        diffX, diffY = np.diff(self._X), np.diff(self._Y)

        if (all(diffX >= 0) or all(diffX <= 0)) and (all(diffY >= 0) or all(diffY <= 0)) and self._k in (1, 3):
            r.criticalPoints = False
        elif self._k == 1:
            r._interval = lambda *args: spline_interval_analysis_engine(r, *args)
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
        self._X, self._Y = np.asfarray(args[0]), np.asfarray(args[1])
        diffY = np.diff(self._Y)
        ind_nonmonotone = np.where(diffY[1:] * diffY[:-1] < 0)[0] + 1
        self._nonmonotone_x = self._X[ind_nonmonotone]
        self._nonmonotone_y = self._Y[ind_nonmonotone]
        
        if len(args) >= 5:
            k = args[4]
        elif 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 3 # default for InterpolatedUnivariateSpline
            
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

def spline_interval_analysis_engine(S, domain, dtype):
    lb_ub, definiteRange = S.input[0]._interval(domain, dtype)
    lb, ub = lb_ub[0], lb_ub[1]
    
    x, y = S._nonmonotone_x, S._nonmonotone_y
    tmp = S.fun(lb_ub)
    _inf, _sup = tmp[0], tmp[1]
    for i, xx in enumerate(x):
        yy = y[i]
        ind = np.logical_and(lb < xx, xx < ub)
        _inf[ind] = np.where(_inf[ind] < yy, _inf[ind], yy)
        _sup[ind] = np.where(_sup[ind] > yy, _sup[ind], yy)
    r = np.vstack((_inf, _sup))

    # TODO: modify definiteRange for out-of-bounds splines
    # definiteRange = False
        
    return r, definiteRange
