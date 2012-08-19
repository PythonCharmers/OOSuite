from baseProblem import NonLinProblem


class ODE(NonLinProblem):
    probType = 'ODE'
    goal = 'solution'
    allowedGoals = ['solution']
    showGoal = False
    _optionalData = []
    FuncDesignerSign = 'timeVariable'
    expectedArgs = ['equations', 'startPoint', 'timeVariable', 'times']
    ftol = None
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        domain, timeVariable, times = args[1:4]
        self.x0 = domain
#        if any(diff(times) < 0): self.err('''
#        currently required ODE times should be sorted 
#        in ascending order, other cases are unimplemented yet
#        ''')
        
        #self.constraints = [timeVariable > times[0], timeVariable < times[-1]]
        

    def objFunc(self, x):
        return 0
        #raise 'unimplemented yet'
        
        #r = norm(dot(self.C, x) - self.d) ** 2  /  2.0
        #return r
