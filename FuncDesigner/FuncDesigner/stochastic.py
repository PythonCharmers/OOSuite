from numpy import asanyarray, ndarray, isscalar, asfarray
from overloads import sum as fdsum
import overloads as o
import operator
from FuncDesigner.ooFun import oofun as oof, FuncDesignerException
from ooarray import ooarray

f_none = lambda *args, **kw: None
class discrete(oof):
    __array_priority__ = 50
    is_var = True
    
    def __init__(self, values, probabilities=None):
        oof.__init__(self, f_none)
        self.stochDep = {self: 1}
        if probabilities is None:
            assert 0, 'this assignment is unimplemented yet'
#            r = asanyarray(values)
#            assert r.ndim == 2
#            assert r.shape[1] == 2
#            self.values, self.probabilities = r[:, 0], r[:, 1]
        else:
            self.values, self.probabilities = asanyarray(values), asanyarray(probabilities)
        
    def _mean(self):
        return fdsum(self.values * self.probabilities)
        
    def _variance(self):
        # may be less than zero due to roundoff errors and thus yield nan in std
        #Tmp = (fdsum((self.values)**2 * self.probabilities) - self.M**2, 0)
        #return o.max(Tmp)
        return o.abs(fdsum((self.values)**2 * self.probabilities) - self.M**2)
        
    def _std(self):
        return o.sqrt(self.Var, attachConstraints = False)
    
    def __getattr__(self, attr):
        if attr == 'M':
            self.M = self._mean()
            return self.M
        elif attr == 'Var':
            self.Var = self._variance()
            return self.Var
        elif attr == 'std':
            self.std = self._std()
            return self.std
        else:
            raise AttributeError('incorrect attribute for FuncDesigner stochastic class')

    def __neg__(self): 
        r = discrete(-self.values, self.probabilities)
        r.stochDep = self.stochDep.copy()
        return r
    __sub__ = lambda self, other: mergeDistributions(self, other, operator.sub)#self + (-asfarray(other).copy()) if type(other) in (list, tuple, ndarray) else self + (-other)

    __add__ = lambda self, other: mergeDistributions(self, other, operator.add)
   
    __mul__ = lambda self, other: mergeDistributions(self, other, operator.mul)
    __rmul__ = lambda self, other: self.__mul__(other)
    
    __div__ = lambda self, other: mergeDistributions(self, other, operator.truediv)    
    __truediv__ = __div__
    
    __pow__ = lambda self, other: mergeDistributions(self, other, operator.pow)
    # TODO: __rpow__
    # TODO: __rdiv__
    def __xor__(self, other): raise FuncDesignerException('For power of oofuns use a**b, not a^b')
        
    def __rxor__(self, other): raise FuncDesignerException('For power of oofuns use a**b, not a^b')    

def mergeDistributions(d1, d2, operation):
    assert isinstance(d1, discrete), 'unimplemented yet'
    cond_d2_nonstoch = not isinstance(d2, discrete)
    cond_same_stoch = not cond_d2_nonstoch and d1.stochDep == d2.stochDep
    if cond_d2_nonstoch or cond_same_stoch:
        if cond_d2_nonstoch:
            d2 = asfarray(d2) if operation == operator.div and not isinstance(d2, ooarray) else asanyarray(d2)
            _vals = d2.reshape(1, -1) if d2.size > 1 else d2
        else:#cond_same_stoch
            _vals = d2.values
        Vals = operation(d1.values, _vals) 
        r = discrete(Vals.flatten(), d1.probabilities.flatten())
        r.stochDep = d1.stochDep.copy()
        return r

    Vals = operation(d1.values.reshape(-1, 1), 
                     (d2.values if operation != operator.truediv \
                     or isinstance(d2.values, (oof, ooarray)) \
                     or isinstance(d1.values, (oof, ooarray))  \
                     else asfarray(d2.values)\
                     ).reshape(1, -1))
    Probabilities = d1.probabilities.reshape(-1, 1) * d2.probabilities.reshape(1, -1)
    r = discrete(Vals.flatten(), Probabilities.flatten())
    r.is_var = False
    
    # adjust stochDep
    stochDep = d1.stochDep.copy()
    for key, val in d2.stochDep.items():
        if key in stochDep:
            raise FuncDesignerException('This SP has structure that makes it impossible to solve in OpenOpt yet')
            stochDep[key] += val
        else:
            stochDep[key] = val
    r.stochDep = stochDep
    return r

from numpy import argsort, cumsum, searchsorted, linspace

def reduce_distrib(values, probabilities, N = 500):
    if len(values) <= N:
        return values, probabilities
    ind = argsort(values)
    values, probabilities = values[ind], probabilities[ind]
    csp = cumsum(probabilities)
    tmp = linspace(0, 1, N)
    Ind = searchsorted(csp, tmp)
    
    # TODO: rework it as linear 1st order approximation
#    from numpy import where
#    J = where(Ind == values.size)[0]
#    print J.size
    Ind[Ind == values.size] -= 1
    new_values = values[Ind] 
    return new_values, asarray([1.0/N]*N)

#from numpy import *
#n = 20
#values = sin(arange(n))
#probabilities = (2 + cos(arange(n))) / sum(2+cos(arange(n)))
#nv,np = reduce_distrib(values,probabilities, 10)
#from pylab import plot, show
#plot(nv)
#show()
