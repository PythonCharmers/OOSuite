from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, all, isfinite, eye, vstack, hstack
from numpy.linalg import norm
from oologfcn import OpenOptException
import NLP

class SLE(MatrixProblem):
    #__optionalData__ = ['damp', 'X', 'c']
    expectedArgs = ['C', 'd']# for FD it should be Cd and x0
    probType = 'SLE'
    goal = 'solution'
    allowedGoals = ['solution']
    showGoal = False
    FuncDesignerSign = 'C'
    solver = 'defaultSLEsolver'
    __optionalData__ = []
    #damp = 0
    
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
                
        #if 'damp' not in kwargs.keys(): self.damp = None
        #if 'f' not in kwargs.keys(): self.f = None
        

    def objFunc(self, x):
        r = norm(dot(self.C, x) - self.d, inf)
#        if self.damp is not None:
#            r += self.damp * norm(x-self.X, inf)
        #if self.f is not None: r += dot(self.f, x)
        return r

#    def llsp2nlp(self, solver, **solver_params):
#        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff, d2f=d2ff)
#        else: p = NLP.NLP(ff, zeros(self.n), df=dff, d2f=d2ff)
#        p.args.f = self # DO NOT USE p.args = self IN PROB ASSIGNMENT!
#        self.inspire(p)
#        self.iprint = -1
#        # for LLSP plot is via NLP
#        p.show = self.show
#        p.plot, self.plot = self.plot, 0
#        #p.checkdf()
#        r = p.solve(solver, **solver_params)
#        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf
#        return r

    def __prepare__(self):
        if isinstance(self.d, dict): # FuncDesigner startPoint 
            self.x0 = self.d
        MatrixProblem.__prepare__(self)
        if self.namedVariablesStyle:
            equations = self.C
            ConstraintTags = [elem.isConstraint for elem in equations]
            cond_all_oofuns_but_not_cons = not any(ConstraintTags) 
            cond_cons = all(ConstraintTags) 
            #print 'cond_all_oofuns_but_not_cons:', cond_all_oofuns_but_not_cons
            #print 'cond_cons:', cond_cons
            if not cond_all_oofuns_but_not_cons and not cond_cons:
                raise OpenOptException('for FuncDesigner sle constructor args must be either all-equalities or all-oofuns')            
                
            C, d = [], []
            Z = self._vector2point(zeros(self.n))
            for elem in self.C:
                if elem.isConstraint:
                    lin_oofun = elem.oofun
                else:
                    lin_oofun = elem
                if not lin_oofun.is_linear:
                    raise OpenOptException('SLE constructor requires all equations to be linear')
                C.append(self._pointDerivative2array(lin_oofun._D(Z, **self._D_kwargs)))
                d.append(-lin_oofun(Z))
            self.C, self.d = vstack(C), hstack(d).flatten()
        self.x0 = zeros(self.C.shape[1])
#        if not self.damp is None and not any(isfinite(self.X)):
#            self.X = zeros(self.n)


#ff = lambda x, LLSPprob: LLSPprob.objFunc(x)
#def dff(x, LLSPprob):
#    r = dot(LLSPprob.C.T, dot(LLSPprob.C,x)  - LLSPprob.d)
#    if not LLSPprob.damp is None: r += LLSPprob.damp*(x - LLSPprob.X)
#    if LLSPprob.f is not None and all(isfinite(LLSPprob.f)) : r += LLSPprob.f
#    return r
#
#def d2ff(x, LLSPprob):
#    r = dot(LLSPprob.C.T, LLSPprob.C)
#    if not LLSPprob.damp is None: r += LLSPprob.damp*eye(x.size)
#    return r
