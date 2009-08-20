from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asarray, ones, inf, array, iterable, sort
from NLP import nlp_init

class MINLP(NonLinProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h', 'discreteVars']
    goal = 'minimum'
    probType = 'MINLP'
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    showGoal = True
    plotOnlyCurrentMinimum = True
    discreteVars = {}
    discrtol = 1e-5 # tolerance required for discrete constraints 
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for MINLP constructor, must be 0..2 + (optionally) some kwargs')
        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['x0'] = args[1]
        NonLinProblem.__init__(self)
        self.iprint=1
        nlp_init(self, kwargs2)

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
        
