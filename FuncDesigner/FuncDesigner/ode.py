from translator import FuncDesignerTranslator
from FDmisc import FuncDesignerException, Extras, _getDiffVarsID
from numpy import ndarray, hstack, vstack, isscalar, asarray, zeros, logical_and, searchsorted
from ooVar import oovar
from ooFun import oofun, atleast_oofun
#from overloads import fixed_oofun

class ode:
    # Ordinary differential equations
    
    _isInitialized = False
    solver = 'scipy_lsoda'
    
    
    def __init__(self, equations, startPoint, timeVariable, times, *args, **kwargs):
        if len(args) > 0:  FuncDesignerException('incorrect ode definition, too many args are obtained')
        
        if not isinstance(equations, dict):
            raise FuncDesignerException('1st argument of ode constructor should be Python dict')

        if not isinstance(startPoint, dict):
            raise FuncDesignerException('2nd argument of ode constructor should be Python dict')
      
        if not isinstance(timeVariable, oovar):
            raise FuncDesignerException('3rd argument of ode constructor should be Python list or numpy array of time values')
        self.timeVariable = timeVariable

        if timeVariable in equations:
            raise FuncDesignerException("ode: differentiation of a variable by itself (time by time) is treated as a potential bug and thus is forbidden")

        if not (isinstance(times, (list, tuple)) or (isinstance(times, ndarray) and times.ndim == 1)): 
            raise FuncDesignerException('4th argument of ode constructor should be Python list or numpy array of time values')
        self.times = times
        self._fd_func, self._startPoint, self._timeVariable, self._times, self._kwargs = equations, startPoint, timeVariable, times,  kwargs
        
        startPoint = dict([(key, val) for key, val in startPoint.items()])
        startPoint[timeVariable] = times[0]
        y0 = []
        Funcs = []
        
        # setting oovar.size is risky - it can affect code written after the ode is solved
        # thus Point4TranslatorAssignment is used instead
        Point4TranslatorAssignment = {}
        
        for v, func in equations.items():
            func = atleast_oofun(func)
#            if not isinstance(func,  oofun):
#                func = fixed_oofun(func)
            if not isinstance(v, oovar):
                raise FuncDesignerException('ode: dict keys must be FuncDesigner oovars, got "%s" instead' % type(v))
            startFVal = asarray(func(startPoint))
            y0.append(asarray(startPoint[v]))
            Funcs.append(func)
            Point4TranslatorAssignment[v] = startFVal
            
            if startFVal.size != asarray(startPoint[v]).size or (hasattr(v, 'size') and isscalar(v.size) and startFVal.size != v.size):
                raise FuncDesignerException('error in user-defined data: oovar "%s" size is not equal to related function value in start point' % v.name)
        
        self.y0 = hstack(y0)
        self.varSizes = [y.size for y in y0]
        ooT = FuncDesignerTranslator(Point4TranslatorAssignment)
        self.ooT = ooT
        def func (y, t): 
            tmp = dict(ooT.vector2point(y))
            tmp[timeVariable] = t
            return hstack([func(tmp) for func in Funcs])
        self.func = func
        
        
        _FDVarsID = _getDiffVarsID()
        def derivative(y, t):
            tmp = dict(ooT.vector2point(y))
            tmp[timeVariable] = t
            r = []
            for func in Funcs:
                tt = func.D(tmp, fixedVarsScheduleID = _FDVarsID)
                tt.pop(timeVariable)
                r.append(ooT.pointDerivative2array(tt))
            return vstack(r)
        self.derivative = derivative
        self.Point4TranslatorAssignment = Point4TranslatorAssignment
        
    def solve(self, solver='scipy_lsoda', *args, **kwargs): # mb for future implementation - add  some **kwargs here as well
        if len(args) > 0:
            raise FuncDesignerException('no args are currently available for the function ode::solve')
        solverName = solver if isinstance(solver, str) else solver.__name__
        if solverName.startswith('interalg'):
            try:
                from openopt import ODE
            except ImportError:
                raise FuncDesignerException('You should have openopt insalled')
            prob = ODE(self._fd_func, self._startPoint, self._timeVariable, self._times, **self._kwargs)
            r = prob.solve(solver, **kwargs)
            y_var = list(prob._x0.keys())[0]
            res = 0.5 * (prob.extras[y_var]['infinums'] + prob.extras[y_var]['supremums'])
            times = 0.5 * (prob.extras['startTimes'] + prob.extras['endTimes'])
            if len(self._times) != 2:
                # old
                from scipy.interpolate import UnivariateSpline
                if 'ftol' in kwargs.keys():
                    s = self._kwargs['ftol']
                elif 'fTol' in kwargs.keys():
                    s = self._kwargs['fTol']
                elif 'ftol' in self._kwargs.keys():
                    s = self._kwargs['ftol']
                elif 'fTol' in self._kwargs.keys():
                    s = self._kwargs['fTol']
                    
                # walkaround a bug with essential slowdown
                if times[-1] < times[0]:
                    times = times[::-1]
                    res = res[::-1]
                    
                interp = UnivariateSpline(times, res, k=1, s=s**2) 
                times = self._times
                res = interp(times)

                # new
#                ind = searchsorted(prob.extras['startTimes'], self._times, 'right')
#                ind[ind==times.size] = times.size-1
#                tmp = res[ind]
##                lti, rti = self._times - prob.extras['startTimes'][ind], prob.extras['endTimes'][ind] - self._times
##                tmp = (lti * res[ind] + rti * res[ind]) / (lti + rti)
##                tmp[logical_and(lti==0, rti==0)] = res[ind]
#                res = tmp
                
            r.xf = {y_var:res, self._timeVariable: times}
            r._xf = {y_var.name: res, self._timeVariable.name: times}
        else:
            if solver != 'scipy_lsoda': raise  FuncDesignerException('incorrect ODE solver')
            try:
                from scipy import integrate
            except ImportError:
                raise FuncDesignerException('to solve ode you mush have scipy installed, see http://openop.org/SciPy')
            y, infodict = integrate.odeint(self.func, self.y0, self.times, Dfun = self.derivative, full_output=True)
            resultDict = dict(self.ooT.vector2point(y.T))
            
            for key, value in resultDict.items():
                if min(value.shape) == 1:
                    resultDict[key] = value.flatten()
            r = FuncDesigner_ODE_Result(resultDict)
            r.msg = infodict['message']
            r.extras = {'infodict': infodict}
        return r
        

class FuncDesigner_ODE_Result:
    # TODO: prevent code clone with runprobsolver.py
    def __init__(self, resultDict):
        self.xf = resultDict
        if not hasattr(self, '_xf'):
            self._xf = dict([(var.name, value) for var, value in resultDict.items()])
        def c(*args):
            r = [(self._xf[arg] if isinstance(arg,  str) else self.xf[arg]) for arg in args]
            return r[0] if len(args)==1 else r
        self.__call__ = c
    pass
