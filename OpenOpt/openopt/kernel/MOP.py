from baseProblem import NonLinProblem
from numpy import inf

from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
class MOP(NonLinProblem):
    _optionalData = ['lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'c', 'h']
    showGoal = True
    goal = 'weak pareto front'
    probType = 'MOP'
    allowedGoals = ['weak pareto front', 'strong pareto front', 'wpf', 'spf']
    isObjFunValueASingleNumber = False
    expectedArgs = ['f', 'x0']
    _frontLength = 0
    _nIncome = 0
    _nOutcome = 0
    __isIterPointAlwaysFeasible__ = True
    iprint = 1
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        self.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        self.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        self.data4TextOutput= ['front length', 'income', 'outcome']
        f = self.f
        i = 0
        targets = []
        while True:
            if len(f[i:]) == 0: break
            func = f[i]
            if type(func) in (list, tuple):
                F, tol, val = func
                i += 1
            else:
                F, tol, val = f[i], f[i+1], f[i+2]
                i += 3
            t = target()
            t.func, t.tol = F, tol
            t.val = val if type(val) != str \
            else inf if val in ('max', 'maximum') \
            else -inf if val in ('min', 'minimum') \
            else self.err('incorrect MOP func target')
            targets.append(t)
        self.targets = targets
        self.f = [t.func for t in targets]
        self.user.f = self.f

    def objFuncMultiple2Single(self, fv):
        return 0#(fv ** 2).sum()

    def solve(self, *args, **kw):
        if self.plot or kw.get('plot', False):
            self.warn('\ninteractive graphic output for MOP is unimplemented yet and will be turned off')
            kw['plot'] = False
        r = NonLinProblem.solve(self, *args, **kw)
        r.plot = lambda *args, **kw: self._plot(**kw)
        return r

    def _plot(self, **kw):
        try:
            import pylab
        except:
            self.err('you should have matplotlib installed')
        if self.nf != 2:
            self.err('MOP plotting is implemented for problems with only 2 goals, while you have %d' % self.nf)
        from numpy import asarray, atleast_1d
        tmp = asarray(self.solutions.F)
        X, Y = atleast_1d(tmp[:, 0]), atleast_1d(tmp[:, 1])
        from copy import deepcopy
        kw2 = deepcopy(kw)
        useGrid = kw2.pop('grid', 'on')
        useShow = kw2.pop('show', True)
        pylab.scatter(X, Y, **kw2)
        pylab.grid(useGrid)
        pylab.xlabel(self.user.f[0].name)
        pylab.ylabel(self.user.f[1].name)
        if useShow: pylab.show()

class target:
    pass
