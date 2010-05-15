from translator import FuncDesignerTranslator
from misc import FuncDesignerException, Extras
from numpy import ndarray, hstack, vstack, isscalar, asarray, zeros
from ooVar import oovar
from ooFun import oofun 
from overloads import fixed_oofun

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
        self.timeVariable = timeVariable

        if timeVariable in equations:
            raise FuncDesignerException("ode: differentiation of a variable by itself (time by time) is treated as a potential bug and thus is forbidden")

        if not (isinstance(timeArray, list) or (isinstance(timeArray, ndarray) and timeArray.ndim == 1)): 
            raise FuncDesignerException('4th argument of ode constructor should be Python list or numpy array of time values')
        self.timeArray = timeArray

        startPoint[timeVariable] = timeArray[0]
        y0 = []
        Funcs = []
        Variables = []
        
        # setting oovar.size is risky - it can affect code written after the ode is solved
        # thus Point4TranslatorAssignment is used instead
        Point4TranslatorAssignment = {timeVariable: timeArray[0]}
        
        for v, func in equations.items():
            Variables.append(v)
            if not isinstance(func,  oofun):
                func = fixed_oofun(func)
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
        self.Variables = Variables
        ooT = FuncDesignerTranslator(Point4TranslatorAssignment)
        self.ooT = ooT
        self.func = lambda y, t: hstack([func(ooT.vector2point(hstack((y, t)))) for func in Funcs])
        self.derivative = lambda y, t: vstack([ooT.pointDerivative2array(func.D(ooT.vector2point(hstack((y, t))))) for func in Funcs])
        self.Point4TranslatorAssignment = Point4TranslatorAssignment
        #self.decode = ooT.vector2point
        
        
        
    def solve(self, *args): # mb for future implementation - add  some **kwargs here as well
        if len(args) > 0:
            raise FuncDesignerException('no args are currently available for the function ode::solve')
        try:
            from scipy import integrate
        except:
            raise FuncDesignerException('to solve ode you mush have scipy installed, see scipy.org')
        y, infodict = integrate.odeint(self.func, self.y0, self.timeArray, Dfun = self.derivative, full_output=True)
        y = y.T
        resultDict = {}
        ac = 0
        for i, v in enumerate(self.Variables):
            resultDict[v] = y[ac:ac+self.varSizes[i]]
            ac += self.varSizes[i]
        #y = hstack((y, zeros((len(self.timeArray), 1))))
        
        #resultDict = dict(self.ooT.vector2point(y.T))
        
#        for key in self.equations.keys():
#            size = self.Point4TranslatorAssignment
#        resultDict = dict([key,])
        
        for key, value in resultDict.items():
            if min(value.shape) == 1:
                resultDict[key] = value.flatten()
        r = FuncDesigner_ODE_Result(resultDict)
        r.msg = infodict['message']
        r.extras = Extras()
        r.extras.infodict = infodict
        return r
        #return y, infodict


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
