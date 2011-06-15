from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, all, isfinite, eye, hstack, vstack, asarray, atleast_2d
from numpy.linalg import norm
import LP

class IP(NonLinProblem):
    probType = 'IP'
    goal = 'solution'
    allowedGoals = ['solution']
    showGoal = False
    _optionalData = []
    expectedArgs = ['f', 'domain']
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        f, domain = args[:2]
        self.x0 = dict([(v, 0.5*(val[0]+val[1])) for v, val in domain.items()])
        self.constraints = [v>bounds[0] for v,  bounds in domain.items()] + [v<bounds[1] for v,  bounds in domain.items()]
        

    def objFunc(self, x):
        return 0
        #raise 'unimplemented yet'
        
        #r = norm(dot(self.C, x) - self.d) ** 2  /  2.0
        #return r
