from baseProblem import NonLinProblem
from numpy import inf

from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
class MOP(NonLinProblem):
    _optionalData = ['lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'c', 'h']
    showGoal = True
    goal = 'weak Pareto front'
    probType = 'MOP'
    allowedGoals = ['weak Pareto front', 'strong Pareto front', 'wpf', 'spf']
    isObjFunValueASingleNumber = False
    expectedArgs = ['f', 'x0']
    _frontLength = 0
    _nIncome = 0
    _nOutcome = 0
    
    iprint = 1
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        self.nSolutions = 'all'
        self.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        self.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        self.data4TextOutput = ['front length', 'income', 'outcome', 'log10(maxResidual)']
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
#        if self.plot or kw.get('plot', False):
#            self.warn('\ninteractive graphic output for MOP is unimplemented yet and will be turned off')
#            kw['plot'] = False
        self.graphics.drawFuncs = [mop_iterdraw]
        r = NonLinProblem.solve(self, *args, **kw)
        r.plot = lambda *args, **kw: self._plot(**kw)
        r.__call__ = lambda *args, **kw: self.err('evaluation of MOP result on arguments is unimplemented yet, use r.solutions')
        return r

    def _plot(self, **kw):
        from numpy import asarray, atleast_1d, array_equal
        S = self.solutions
        if type(S)==list and len(S) == 0: return
        tmp = asarray(self.solutions.F if 'F' in dir(self.solutions) else self.solutions.values)
        from copy import deepcopy
        kw2 = deepcopy(kw)
        useShow = kw2.pop('show', True)
        if not useShow and hasattr(self, '_prev_mop_solutions') and array_equal(self._prev_mop_solutions, tmp):
            return
        self._prev_mop_solutions = tmp.copy()
        if tmp.size == 0:
            self.disp('no solutions, nothing to plot')
            return
        try:
            import pylab
        except:
            self.err('you should have matplotlib installed')
        pylab.ion()
        if self.nf != 2:
            self.err('MOP plotting is implemented for problems with only 2 goals, while you have %d' % self.nf)
        X, Y = atleast_1d(tmp[:, 0]), atleast_1d(tmp[:, 1])

        useGrid = kw2.pop('grid', 'on')
        
        if 'marker' not in kw2: 
            kw2['marker'] = (5, 1, 0)
        if 's' not in kw2:
            kw2['s']=[150]
        if 'edgecolor' not in kw2:
            kw2['edgecolor'] = 'b'
        if 'facecolor' not in kw2:
            kw2['facecolor'] = '#FFFF00'#'y'
            
        pylab.scatter(X, Y, **kw2)
        
        pylab.grid(useGrid)
        t0_goal = 'min' if self.targets[0].val == -inf else 'max' if self.targets[0].val == inf else str(self.targets[0].val)
        t1_goal = 'min' if self.targets[1].val == -inf else 'max' if self.targets[1].val == inf else str(self.targets[1].val)
        
        pylab.xlabel(self.user.f[0].name + ' (goal: %s    tolerance: %s)' %(t0_goal, self.targets[0].tol))
        pylab.ylabel(self.user.f[1].name + ' (goal: %s    tolerance: %s)' %(t1_goal, self.targets[1].tol))
        
        pylab.title('problem: %s    goal: %s' %(self.name, self.goal))
        figure = pylab.gcf()
        from openopt import __version__ as ooversion
        figure.canvas.set_window_title('OpenOpt ' + ooversion)
        
        pylab.hold(0)
        pylab.draw()
        if useShow: 
            pylab.ioff()
            pylab.show()

def mop_iterdraw(p):
    useShow = False#p.isFinished and p.show
    p._plot(show=useShow)

class target:
    pass
