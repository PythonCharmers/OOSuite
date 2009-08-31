##these 2 lines are for openopt developers ONLY!
import sys, os.path as pth
sys.path.insert(0,pth.split(pth.split(pth.split(pth.split(pth.realpath(pth.dirname(__file__)))[0])[0])[0])[0])
###############################
#import NLP

from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, asfarray, nan, zeros, isfinite, all


class SDP(MatrixProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'S', 'd']
    expectedArgs = ['f']
    goal = 'minimum'
    #TODO: impolement goal = max, maximum for SDP
    #allowedGoals = ['minimum', 'min', 'maximum', 'max']
    allowedGoals = ['minimum', 'min']
    showGoal = True    
    def __init__(self, *args, **kwargs):
        self.probType = 'SDP'
        self.S = {}
        self.d = {}
        MatrixProblem.__init__(self, *args, **kwargs)
        self.f = asfarray(self.f)
        self.n = self.f.size
        if self.x0 is None: self.x0 = zeros(self.n)
        
    def __prepare__(self):
        MatrixProblem.__prepare__(self)
        if self.solver.__name__ in ['cvxopt_sdp', 'dsdp']:
            try:
                from cvxopt.base import matrix
                matrixConverter = lambda x: matrix(x, tc='d')
            except:
                self.err('cvxopt must be installed')
        else:
            matrixConverter = asfarray
        for i in self.S.keys(): self.S[i] = matrixConverter(self.S[i])
        for i in self.d.keys(): self.d[i] = matrixConverter(self.d[i])
#        if len(S) != len(d): self.err('semidefinite constraints S and d should have same length, got '+len(S) + ' vs '+len(d)+' instead')
#        for i in xrange(len(S)):
#            d[i] = matrixConverter(d[i])
#            for j in xrange(len(S[i])):
#                S[i][j] = matrixConverter(S[i][j])
            
        
        
    def __finalize__(self):
        MatrixProblem.__finalize__(self)
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
                if hasattr(self, fn):
                    setattr(self, fn, -getattr(self, fn))
        

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

