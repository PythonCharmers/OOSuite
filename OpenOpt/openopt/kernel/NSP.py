from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asarray, ones, inf

class NSP(NonLinProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    expectedArgs = ['f', 'x0']
    goal = 'minimum'
    probType = 'NSP'
    JacobianApproximationStencil = 2
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    showGoal = True    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
       #TODO: set here default tolx, tolcon, diffInt etc for NS Problem







