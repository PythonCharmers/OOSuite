from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asarray, ones, inf, array, iterable, sort

class MINLP(NonLinProblem):
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h', 'discreteVars']
    goal = 'minimum'
    probType = 'MINLP'
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    showGoal = True
    plotOnlyCurrentMinimum = True
    discreteVars = {}
    discrtol = 1e-5 # tolerance required for discrete constraints 
    expectedArgs = ['f', 'x0']
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        self.iprint=1

    def __prepare__(self):
        if hasattr(self, 'prepared') and self.prepared == True:
            return
        NonLinProblem.__prepare__(self)    
        # TODO: use something else instead of dict.keys()
        for key in self.discreteVars.keys():
            fv = self.discreteVars[key]
            if type(fv) not in [list, tuple]:
                self.err('each element from discreteVars dictionary should be list or tuple of allowed values')
            fv = sort(fv)
            if fv[0]>self.ub[key]:
                self.err('variable '+ str(key)+ ': smallest allowed discrete value ' + str(fv[0]) + ' exeeds imposed upper bound '+ str(self.ub[key]))
            if fv[-1]<self.lb[key]:
                self.err('variable '+ str(key)+ ': biggest allowed discrete value ' + str(fv[-1]) + ' is less than imposed lower bound '+ str(self.lb[key]))
            self.discreteVars[key] = fv
        
