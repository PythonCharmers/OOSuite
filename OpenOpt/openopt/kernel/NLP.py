from ooMisc import assignScript
from baseProblem import NonLinProblem
from numpy import asarray, ones, inf, array, iterable


class NLP(NonLinProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    goal = 'minimum'
    probType = 'NLP'
    allowedGoals = ['minimum', 'min', 'maximum', 'max']
    showGoal = True
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for NLP constructor, must be 0..2 + (optionaly) some kwargs')
        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['x0'] = args[1]
        NonLinProblem.__init__(self)
        nlp_init(self, kwargs2)


def nlp_init(prob, kwargs):
    f = kwargs['f']

    if isinstance(f, basestring):
        prob.err("Isn't implemented yet")
        # TODO: implement me!
        # prob. f, prob.fName = ..., f
    elif hasattr(f, '__name__'):
        prob.fName = f.__name__
    elif hasattr(f, 'name'):
        prob.fName = f.name
    elif iterable(f):
        prob.fName = "undefined"
    elif not callable(f): prob.err('incorrect objFun')
    prob.f = f
    return assignScript(prob, kwargs)
