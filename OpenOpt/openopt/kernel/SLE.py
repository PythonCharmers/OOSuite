from ooMisc import assignScript
from baseProblem import MatrixProblem
from numpy import asfarray, ones, inf, dot, nan, zeros, any, all, isfinite, eye, vstack, hstack, flatnonzero, isscalar, ndarray, atleast_2d
from numpy.linalg import norm
from oologfcn import OpenOptException
import NLP

try:
    import scipy
    scipyInstalled = True
except:
    scipyInstalled = False

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
    _isPrepared = False
    
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)
    
    def asSparse(self):
        return True if self.n > 100 else False

    def objFunc(self, x):
        if isinstance(self.C, ndarray):
            return norm(dot(self.C, x) - self.d, inf)
        else:
            # TODO: omit code clone in FD ooFun.py, function _D
            #print '1'
            t1 = self.C_as_csr
            #print '2'
            t2 = scipy.sparse.csc_matrix(x)
            #print '3'
            if t2.shape[0] != t1.shape[1]:
                if t2.shape[1] == t1.shape[1]:
                    t2 = t2.T
                else:
                    raise FuncDesignerException('incorrect shape in FuncDesigner function _D(), inform developers about the bug')
            #print '4'
            rr =  t1._mul_sparse_matrix(t2)            
            #print '5'
            r = norm(rr.todense().A - self.d, inf)
            #print '6'
            return r

    def __prepare__(self):
        if self._isPrepared: return
        self._isPrepared = True
        if isinstance(self.d, dict): # FuncDesigner startPoint 
            self.x0 = self.d
        MatrixProblem.__prepare__(self)
        if self.isFDmodel:
            equations = self.C
            ConstraintTags = [elem.isConstraint for elem in equations]
            cond_all_oofuns_but_not_cons = not any(ConstraintTags) 
            cond_cons = all(ConstraintTags) 
            #print 'cond_all_oofuns_but_not_cons:', cond_all_oofuns_but_not_cons
            #print 'cond_cons:', cond_cons
            if not cond_all_oofuns_but_not_cons and not cond_cons:
                raise OpenOptException('for FuncDesigner sle constructor args must be either all-equalities or all-oofuns')            
            
            AsSparse = self.asSparse if isscalar(self.asSparse) else self.asSparse()
            
            C, d = [], []
            Z = self._vector2point(zeros(self.n))
            for elem in self.C:
                if elem.isConstraint:
                    lin_oofun = elem.oofun
                else:
                    lin_oofun = elem
                if not lin_oofun.is_linear:
                    raise OpenOptException('SLE constructor requires all equations to be linear')
                C.append(self._pointDerivative2array(lin_oofun._D(Z, **self._D_kwargs), asSparse = AsSparse))
                d.append(-lin_oofun(Z))
                
            if AsSparse:
                Vstack = scipy.sparse.vstack
            else:
                Vstack = vstack # i.e. numpy.vstack
            #raise 0
            self.C, self.d = Vstack(C), hstack(d).flatten()
            if AsSparse: self.C_as_csr = self.C.tocsr()
            
            if isinstance(self.C,ndarray) and self.n > 100 and len(flatnonzero(self.C))/self.C.size < 0.3:
                s = "Probably you'd better solve this SLE as sparse"
                if not scipyInstalled: s += ' (requires scipy installed)'
                self.pWarn(s)

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
