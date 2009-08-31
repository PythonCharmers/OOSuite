##these 2 lines are for openopt developers ONLY!
import sys, os.path as pth
sys.path.insert(0,pth.split(pth.split(pth.split(pth.split(pth.realpath(pth.dirname(__file__)))[0])[0])[0])[0])
###############################
#import NLP

from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, asfarray, nan, zeros, isfinite, all


class SOCP(MatrixProblem):
    probType = 'SOCP'
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    goal = 'minimum'
    allowedGoals = ['minimum', 'min']
    #TODO: add goal=max, maximum
    showGoal = True
    expectedArgs = ['f', 'C', 'd']
    # required are f, C, d
    
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        self.f = asfarray(self.f)
        self.n = self.f.size
        if self.x0 is None: self.x0 = zeros(self.n)


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






#ff = lambda x, QProb: QProb.objFunc(x)
#def dff(x, QProb):
#    r = dot(QProb.H, x)
#    if all(isfinite(QProb.f)) : r += QProb.f
#    return r
#
#def d2ff(x, QProb):
#    r = QProb.H
#    return r

