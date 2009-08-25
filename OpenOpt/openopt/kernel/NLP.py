from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asarray, ones, inf, array, iterable


class NLP(NonLinProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    expectedArgs = ['f', 'x0']
    goal = 'minimum'
    probType = 'NLP'
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    showGoal = True
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
