from numpy import copy
from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, nan, zeros, ceil, floor, argmax
from setDefaultIterFuncs import SMALL_DELTA_X, SMALL_DELTA_F
from LP import LP



class MILP(LP):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'intVars', 'boolVars']
    storeIterPoints = False
    probType = 'MILP'
    expectedArgs = ['f', 'x0']
    goal = 'minimum'
    allowedGoals = ['minimum', 'min', 'max', 'maximum']
    showGoal = True
    
    def __prepare__(self):
        LP.__prepare__(self)
        r = []
        if type(self.intVars) not in [list, tuple]:
            self.intVars = [self.intVars]
        if self.isFDmodel:
            for iv in self.intVars:
                if self.fixedVars is not None and iv in self.fixedVars or\
                self.optVars is not None and iv not in self.optVars:
                    continue
                r1, r2 = self._oovarsIndDict[iv.name]
                r += range(r1, r2)
            self.intVars, self._intVars = r, self.intVars
                
        if SMALL_DELTA_X in self.kernelIterFuncs: self.kernelIterFuncs.pop(SMALL_DELTA_X)
        if SMALL_DELTA_F in self.kernelIterFuncs: self.kernelIterFuncs.pop(SMALL_DELTA_F)
        def getMaxResidualWithIntegerConstraints(x, retAll = False):
            r, fname, ind = self.getMaxResidual2(x, True)
            intV = x[self.intVars]
            intDifference = abs(intV-intV.round())
            intConstraintNumber = argmax(intDifference)
            intConstraint = intDifference[intConstraintNumber]
            #print 'intConstraint:', intConstraint
            if intConstraint > r:
                intConstraintNumber = self.intVars[intConstraintNumber]
                r, fname, ind = intConstraint, 'int', intConstraintNumber 
            if retAll:
                return r, fname, ind
            else:
                return r
        self.getMaxResidual, self.getMaxResidual2 = getMaxResidualWithIntegerConstraints, self.getMaxResidual
        
        # TODO: 
        # 1) ADD BOOL VARS
        self.lb, self.ub = copy(self.lb), copy(self.ub)
        self.lb[self.intVars] = ceil(self.lb[self.intVars])
        self.ub[self.intVars] = floor(self.ub[self.intVars])
        
#        if self.goal in ['max', 'maximum']:
#            self.f = -self.f
    def __finalize__(self):
        LP.__finalize__(self)
        if self.isFDmodel: self.intVars = self._intVars
#    def __finalize__(self):
#        MatrixProblem.__finalize__(self)
#        if self.goal in ['max', 'maximum']:
#            self.f = -self.f
#            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
#                if hasattr(self, fn):
#                    setattr(self, fn, -getattr(self, fn))
    
#    def __init__(self, *args, **kwargs):
#        LP.__init__(self, *args, **kwargs)

    def objFunc(self, x):
        return dot(self.f, x) + self._c
        #return dot(self.f, x)



