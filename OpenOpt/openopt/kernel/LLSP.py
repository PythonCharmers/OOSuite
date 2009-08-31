from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, all, isfinite, eye
from numpy.linalg import norm
import NLP

class LLSP(MatrixProblem):
    __optionalData__ = ['damp', 'X', 'c']
    expectedArgs = ['C', 'd']
    probType = 'LLSP'
    goal = 'minimum'
    allowedGoals = ['minimum', 'min']
    showGoal = False
    
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
        if len(args)>1:
            self.n = args[0].shape[1]
        else:
            self.n = kwargs['C'].shape[1]
        #self.lb = -inf * ones(self.n)
        #self.ub =  inf * ones(self.n)
        if 'damp' not in kwargs.keys(): self.damp = None
        if 'f' not in kwargs.keys(): self.f = None

        if self.x0 is None: self.x0 = zeros(self.n)        


    def objFunc(self, x):
        r = norm(dot(self.C, x) - self.d) ** 2  /  2.0
        if self.damp is not None:
            r += self.damp * norm(x-self.X)**2 / 2.0
        if self.f is not None: r += dot(self.f, x)
        return r

    def llsp2nlp(self, solver, **solver_params):
        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff, d2f=d2ff)
        else: p = NLP.NLP(ff, zeros(self.n), df=dff, d2f=d2ff)
        p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
        self.inspire(p)
        self.iprint = -1
        # for LLSP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0
        #p.checkdf()
        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
        return r

    def __prepare__(self):
        MatrixProblem.__prepare__(self)
        if not self.damp is None and not any(isfinite(self.X)):
            self.X = zeros(self.n)




#def llsp_init(prob, kwargs):
#    if 'damp' not in kwargs.keys(): kwargs['damp'] = None
#    if 'X' not in kwargs.keys(): kwargs['X'] = nan*ones(prob.n)
#    if 'f' not in kwargs.keys(): kwargs['f'] = nan*ones(prob.n)
#
#    if prob.x0 is nan: prob.x0 = zeros(prob.n)


#def ff(x, LLSPprob):
#    r = dot(LLSPprob.C, x) - LLSPprob.d
#    return dot(r, r)
ff = lambda x, LLSPprob: LLSPprob.objFunc(x)
def dff(x, LLSPprob):
    r = dot(LLSPprob.C.T, dot(LLSPprob.C,x)  - LLSPprob.d)
    if not LLSPprob.damp is None: r += LLSPprob.damp*(x - LLSPprob.X)
    if all(isfinite(LLSPprob.f)) : r += LLSPprob.f
    return r

def d2ff(x, LLSPprob):
    r = dot(LLSPprob.C.T, LLSPprob.C)
    if not LLSPprob.damp is None: r += LLSPprob.damp*eye(x.size)
    return r
