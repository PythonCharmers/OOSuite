from baseProblem import MatrixProblem
from STAB import set_routine

class DSP(MatrixProblem):
    _optionalData = []
    probType = 'DSP'
    expectedArgs = ['graph']
    allowedGoals = ['minimum dominating set']
    showGoal = False

    _init = False
    
    def __setattr__(self, attr, val): 
        if self._init: self.err('openopt DSP instances are immutable, arguments should pass to constructor or solve()')
        self.__dict__[attr] = val

    def __init__(self, *args, **kw):
        MatrixProblem.__init__(self, *args, **kw)
        self.__init_kwargs = kw
        self._init = True
        
    def solve(self, *args, **kw):
        return set_routine(self, *args, **kw)
#        from openopt import STAB
#        KW = self.__init_kwargs
#        KW.update(kw)
#        P = STAB(graph, **KW)
#        r = P.solve(*args)
#        return r
        
