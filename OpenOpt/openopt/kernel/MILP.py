from numpy import copy
from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, nan, zeros, ceil, floor, argmax
from setDefaultIterFuncs import SMALL_DELTA_X, SMALL_DELTA_F
from LP import lp_init



class MILP(MatrixProblem):
    __optionalData__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    storeIterPoints = False
    
    def __prepare__(self):
        MatrixProblem.__prepare__(self)
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
        
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
    
    def __finalize__(self):
        MatrixProblem.__finalize__(self)
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
                if hasattr(self, fn):
                    setattr(self, fn, -getattr(self, fn))
    
    def __init__(self, *args, **kwargs):
        if len(args) > 2: self.err('incorrect args number for MILP constructor, must be 0..2 + (optionaly) some kwargs')

        kwargs2 = kwargs.copy()
        if len(args) > 0: kwargs2['f'] = args[0]
        if len(args) > 1: kwargs2['intVars'] = args[1]
        self.probType = 'MILP'
        MatrixProblem.__init__(self)
        lp_init(self, kwargs2)

    def objFunc(self, x):
        return dot(self.f, x)



