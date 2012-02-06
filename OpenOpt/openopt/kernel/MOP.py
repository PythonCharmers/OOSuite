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
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        self.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        self.kernelIterFuncs.pop(SMALL_DELTA_F, None)
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

class target:
    pass
