##these 2 lines are for openopt developers ONLY!
import sys, os.path as pth
sys.path.insert(0,pth.split(pth.split(pth.split(pth.split(pth.realpath(pth.dirname(__file__)))[0])[0])[0])[0])
###############################
#import NLP

from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, asfarray, nan, zeros, isfinite, all


class SOCP(MatrixProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    # required are f, C, d
    
    def __init__(self, *args, **kwargs):
        self.probType = 'SOCP'
        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['C'] = args[1]
        if len(args) > 2: kwargs2['d'] = args[2]
        if len(args) > 3: self.err('incorrect args number for SOCP constructor, must be 0..2 + (optionaly) some kwargs')
        MatrixProblem.__init__(self)

        return socp_init(self, kwargs2)

    def objFunc(self, x):
        return asfarray(dot(self.f, x).sum()).flatten()

#    def qp2nlp(self, solver, **solver_params):
#        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff, d2f=d2ff)
#        else: p = NLP.NLP(ff, zeros(self.n), df=dff, d2f=d2ff)
#        p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
#        self.inspire(p)
#        self.iprint = -1
#
#        # for QP plot is via NLP
#        p.show = self.show
#        p.plot, self.plot = self.plot, 0
#
#        #p.checkdf()
#        r = p.solve(solver, **solver_params)
#        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
#        return r

def socp_init(p, kwargs):
    p.goal = 'minimum'
    p.allowedGoals = ['minimum', 'min']
    #TODO: add goal=max, maximum
    p.showGoal = True

    for fn in ('f', ):
        if fn in kwargs.keys():
            kwargs[fn] = asfarray(kwargs[fn], float) # TODO: handle the case in runProbSolver()

    p.n = kwargs['f'].size
    if p.x0 is nan: p.x0 = zeros(p.n)
    p.lb = -inf * ones(p.n)
    p.ub =  inf * ones(p.n)

    return assignScript(p, kwargs)

#ff = lambda x, QProb: QProb.objFunc(x)
#def dff(x, QProb):
#    r = dot(QProb.H, x)
#    if all(isfinite(QProb.f)) : r += QProb.f
#    return r
#
#def d2ff(x, QProb):
#    r = QProb.H
#    return r

