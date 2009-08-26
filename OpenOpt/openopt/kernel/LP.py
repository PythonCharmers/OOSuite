from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, nan, zeros, isnan, any
import NLP

class LP(MatrixProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    expectedArgs = ['f', 'x0']
    goal = 'minimum'
    probType = 'LP'
    allowedGoals = ['minimum', 'min', 'max', 'maximum']
    showGoal = True

    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        if len(args) > 1 and not hasattr(args[0], 'is_oovar'):
            self.err('No more than 1 argument is allowed for classic style LP constructor')

    def __prepare__(self):
        MatrixProblem.__prepare__(self)
        if self.x0 is None: self.x0 = zeros(self.n)
        if hasattr(self.f, 'is_oovar'): # hence is oofun or oovar
            _f = self._point2vector(self.f.D(self._x0))
            self.f, self._f = _f, self.f
            _c = self._f(self._x0) - dot(self.f, self.x0)
            self._c = _c
            
        else:
            self._c = 0
        if not hasattr(self, 'n'): self.n = len(self.f)
        #print 'lb:', self.lb, 'ub:', self.ub
        if not hasattr(self, 'lb'): self.lb = -inf * ones(self.n)
        if not hasattr(self, 'ub'): self.ub = inf * ones(self.n)
#        if any(isnan(self.lb)): 
#            if self.lb.size != 1: self.err('NaN in lower bound for a variable from the problem')
#            self.lb = -inf * ones(self.n)
#        if any(isnan(self.ub)): 
#            if self.ub.size != 1: self.err('NaN in upper bound for a variable from the problem')
#            self.ub = inf * ones(self.n)
        
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
            
    # TODO: handle this and SDP finalize in single func finalize_for_max
    def __finalize__(self):
        MatrixProblem.__finalize__(self)
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
                if hasattr(self, fn):
                    setattr(self, fn, -getattr(self, fn))
        if hasattr(self, '_f'):
            self.f = self._f
            self.xf = self._vector2point(self.xf)

            
    def objFunc(self, x):
        return dot(self.f, x) + self._c

    def lp2nlp(self, solver, **solver_params):
        if self.isConverterInvolved and self.goal in ['max', 'maximum']:
            self.err('maximization problems are not implemented lp2nlp converter')
        ff = lambda x: dot(x, self.f)
        dff = lambda x: self.f
        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff)
        else: p = NLP.NLP(ff, zeros(self.n), df=dff)
        self.inspire(p)
        self.iprint = -1

        # for LP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0

        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf

        return r





