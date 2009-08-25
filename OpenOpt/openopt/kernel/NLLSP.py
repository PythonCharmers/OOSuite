from baseProblem import NonLinProblem
from numpy import sum, dot, asfarray
import NLP

class NLLSP(NonLinProblem):
    __optionalData__ = ['lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'c', 'h']
    showGoal = False
    goal = 'minimum'
    probType = 'NLLSP'
    allowedGoals = ['minimum', 'min']
    isObjFunValueASingleNumber = False
    expectedArgs = ['f', 'x0']
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)

    def objFuncMultiple2Single(self, fv):
        return (fv ** 2).sum()

    def nllsp2nlp(self, solver, **solver_params):

        ff = lambda x: sum(asfarray(self.f(x))**2)
        if self.userProvided.df:
            dff = lambda x: dot(2*asfarray(self.f(x)), asfarray(self.df(x)))
            p = NLP.NLP(ff, self.x0, df=dff)
        else:
            p = NLP.NLP(ff, self.x0)
        #p = NLP.NLP(FF, self.x0)
        self.inspire(p, sameConstraints=True)


        def lsp_iterfcn(*args,  **kwargs):
            p.primalIterFcn(*args,  **kwargs)
            p.xk = self.xk
            p.fk = p.f(p.xk)
            p.rk = self.rk
            # TODO: add nNaNs

            p.istop = self.istop

        p.primalIterFcn,  p.iterfcn = self.iterfcn, lsp_iterfcn

        self.iprint = -1
        p.show = False

        r = p.solve(solver, **solver_params)
        #r.ff = ff(r.xf)

        return r
