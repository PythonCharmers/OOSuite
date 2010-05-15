from translator import FuncDesignerTranslator
from misc import FuncDesignerException
from numpy import ndarray, hstack, vstack, isscalar
from ooVar import oovar

class ode:
    # Ordinary differential equations
    
    _isInitialized = False
    solver = 'scipy_lsoda'
    
    def __init__(self, equations, startPoint, timeVariable, timeArray, *args, **kwargs):
        if len(args) > 0:  FuncDesignerException('incorrect ode definition, too many args are obtained')
        
        if not isinstance(equations, dict):
            raise FuncDesignerException('1st argument of ode constructor should be Python dict')

        if not isinstance(startPoint, dict):
            raise FuncDesignerException('2nd argument of ode constructor should be Python dict')
      
        if not isinstance(timeVariable, oovar):
            raise FuncDesignerException('3rd argument of ode constructor should be Python list or numpy array of time values')

        if timeVariable in equations:
            raise FuncDesignerException("ode: differentiation of a variable by itself (time by time) is treated as a potential bug and thus is forbidden")

        if not (isinstance(timeArray, list) or (isinstance(timeArray, ndarray) and timeArray.ndim == 1)): 
            raise FuncDesignerException('4th argument of ode constructor should be Python list or numpy array of time values')
        self.timeArray = timeArray

        startPoint[timeVariable] = timeArray[0]
        y0 = []
        Funcs = []
        
        # setting oovar.size is risky - it can affect code written after the ode is solved
        # thus Point4TranslatorAssignment is used instead
        Point4TranslatorAssignment = {timeVariable: timeArray[0]}
        
        for v, func in equations.items():
            if not isinstance(v, oovar):
                raise FuncDesignerException('ode: dict keys must be FuncDesigner oovars, got "%s" instead' % type(v))
            startFVal = func(startPoint)
            y0.append(startPoint[v])
            Funcs.append(func)
            Point4TranslatorAssignment[v] = startFVal
            
            if hasattr(v, 'size') and isscalar(v.size) and startFVal.size != v.size:
                raise FuncDesignerException('error in user-defined data: oovar "%s" size is not equal to related function value in start point' % v.name)
        
        self.y0 = hstack(y0)
        ooT = FuncDesignerTranslator(Point4TranslatorAssignment)
        self.func = lambda y, t: hstack([func(ooT.vector2point(hstack((y, t)))) for func in Funcs])
#        def f(y, t):
#            
#            return hstack([func(ooT.vector2point(y)) for func in Funcs])
#        self.func = f    
        self.derivative = lambda y, t: vstack([ooT.pointDerivative2array(func.D(ooT.vector2point(hstack(y, t)))) for func in Funcs])
        #self.decode = ooT.vector2point
        
        
        
    def solve(self, *args): # mb for future implementation - add  some **kwargs here as well
        if len(args) > 0:
            raise FuncDesignerException('no args are currently available for the function ode::solve')
        try:
            from scipy import integrate
        except:
            raise FuncDesignerException('to solve ode you mush have scipy installed, see scipy.org')
        y, infodict = integrate.odeint(self.func, self.y0, self.timeArray, Dfun = self.derivative, full_output=True)
        return y, infodict

        #self.decodeArgs(*args)
        #r = self.p.solve(matrixSLEsolver=self.matrixSLEsolver)
#        if r.istop >= 0:
#            return r
#        else:
#            R = {}
#            for key, value in self.p.x0.items(): 
#                R[key] = value * nan
#            r.xf = R
#            r.ff = inf
#        return r
            
#    def decodeArgs(self, *args):
#        hasStartPoint = False
#        for arg in args:
#            if isinstance(arg, str):
#                self.matrixSLEsolver = arg
#            elif isinstance(arg, dict):
#                startPoint = args[0]
#                hasStartPoint = True
#            else:
#                raise FuncDesignerException('incorrect arg type, should be string (solver name) or dict (start point)')
#            
#        if not hasStartPoint:  
#            if hasattr(self, 'startPoint'): return # established from __init__
#            involvedOOVars = set()
#            for Elem in self.equations:
#                elem = Elem.oofun if Elem.isConstraint else Elem
#                if elem.is_oovar:
#                    involvedOOVars.add(elem)
#                else:
#                    involvedOOVars.update(elem._getDep())
#            startPoint = {}
#            for oov in involvedOOVars:
#                if isscalar(oov.size):
#                    startPoint[oov] = zeros(oov.size)
#                else:
#                    startPoint[oov] = 0
#        self.startPoint = startPoint
