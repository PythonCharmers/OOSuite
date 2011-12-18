from baseProblem import MatrixProblem
#from numpy.linalg import norm

class EIG(MatrixProblem):
    probType = 'EIG'
    goal = 'all'
    allowedGoals = None
    showGoal = True
    expectedArgs = ['C']
    M = None
    _optionalData = ['M']
    xtol = 0.0
    FuncDesignerSign = 'C'
    N = 0
    
    #ftol = None
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)

        if self.goal == 'all':
            Name, name = 'all eigenvectors and eigenvalues', 'all'
            N = self.C.shape[0]
        else:
            assert type(self.goal) in (dict, tuple, list) and len(self.goal) == 1, \
            'EIG goal argument should be "all" or Python dict {goal_name: number_of_required_eigenvalues}'
            if type(self.goal) == dict:
                goal_name, N = self.goal.items()[0]
            else:
                goal_name, N = self.goal
            name = ''.join(goal_name.lower().split())
            if name  in ('lm', 'largestmagnitude'):
                Name, name = 'largest magnitude', 'le'
            elif name in ('sm', 'smallestmagnitude'):
                Name, name = 'smallest magnitude', 'sm'
            elif name in ('lr', 'largestrealpart'):
                Name, name = 'largest real part', 'lr'
            elif name in ('sr', 'smallestrealpart'):
                Name, name = 'smallest real part', 'sr'
            elif name in ('li', 'largestimaginarypart'):
                Name, name = 'largest imaginary part', 'li'
            elif name in ('si', 'smallestimaginarypart'):
                Name, name = 'smallest imaginary part', 'si'
            elif name in ('la', 'largestamplitude'):
                Name, name = 'largestamplitude', 'la'
            elif name in ('sa', 'smallestamplitude'):
                Name, name = 'smallest amplitude', 'sa'
            elif name in ('be', 'bothendsofthespectrum'):
                Name, name = 'both ends of the spectrum', 'be'
        
        self.goal = Name
        self._goal = name
        self.N = N

    def objFunc(self, x):
        return 0
        #raise 'unimplemented yet'
        
