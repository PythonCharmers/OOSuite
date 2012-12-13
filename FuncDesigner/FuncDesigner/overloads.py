PythonSum = sum
PythonMax = max
from ooFun import oofun
import numpy as np
from FDmisc import FuncDesignerException, Diag, Eye, raise_except, diagonal, DiagonalType
from ooFun import atleast_oofun, Vstack, Copy
from ooarray import ooarray
from Interval import TrigonometryCriticalPoints, nonnegative_interval, ZeroCriticalPointsInterval, box_1_interval
from numpy import atleast_1d, logical_and
from FuncDesigner.multiarray import multiarray

try:
    from scipy.sparse import isspmatrix, lil_matrix as Zeros
    scipyInstalled = True
except ImportError:
    scipyInstalled = False
    isspmatrix = lambda *args, **kw: False
    Zeros = np.zeros 
    
__all__ = []

#class unary_oofun_overload:
#    def __init__(self, *args, **kwargs):
#        assert len(args) == 1 and len(kwargs) == 0
#        self.altFunc = args[0]
#
#    def __call__(self, *args, **kwargs):
#        assert len(args) == 1 and len(kwargs) == 0
#        if not isinstance(args[0], oofun):
#            return self.altFunc(*args)
#        return self.fun(*args, **kwargs)

#@unary_oofun_overload(np.sin)
#def sin(inp):
#    #if not isinstance(inp, oofun): return np.sin(inp)
#    def d(x):
#        return np.cos(x)
#    return oofun(lambda x: np.sin(x), inp, d = d)

try:
    import distribution
    hasStochastic = True
except:
    hasStochastic = False

#hasStochastic = False

st_sin = (lambda x: \
distribution.stochasticDistribution(sin(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([sin(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.sin(x))\
if hasStochastic\
else np.sin

def sin(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([sin(elem) for elem in inp])
    elif hasStochastic and isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(sin(inp.values), inp.probabilities.copy())._update(inp)
    elif not isinstance(inp, oofun): return np.sin(inp)
    return oofun(st_sin, inp, 
                 d = lambda x: Diag(np.cos(x)), 
                 vectorized = True, 
                 criticalPoints = TrigonometryCriticalPoints)
                 #_interval = lambda domain: ufuncInterval(inp, domain, np.sin, TrigonometryCriticalPoints))

st_cos = (lambda x: \
distribution.stochasticDistribution(cos(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([cos(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.cos(x))\
if hasStochastic\
else np.cos

def cos(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([cos(elem) for elem in inp])
    elif hasStochastic and isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(cos(inp.values), inp.probabilities.copy())._update(inp)        
    elif not isinstance(inp, oofun): return np.cos(inp)
    #return oofun(np.cos, inp, d = lambda x: Diag(-np.sin(x)))
    return oofun(st_cos, inp, 
             d = lambda x: Diag(-np.sin(x)), 
             vectorized = True, 
             criticalPoints = TrigonometryCriticalPoints)
             #_interval = lambda domain: ufuncInterval(inp, domain, np.cos, TrigonometryCriticalPoints))

st_tan = (lambda x: \
distribution.stochasticDistribution(tan(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([tan(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.tan(x))\
if hasStochastic\
else np.tan

def tan(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([tan(elem) for elem in inp])
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(tan(inp.values), inp.probabilities.copy())._update(inp)       
    if not isinstance(inp, oofun): return np.tan(inp)
    # TODO: move it outside of tan definition
    def interval(*args):
        raise 'interval for tan is unimplemented yet'
    r = oofun(st_tan, inp, d = lambda x: Diag(1.0 / np.cos(x) ** 2), vectorized = True, interval = interval)
    return r
    
__all__ += ['sin', 'cos', 'tan']

# TODO: cotan?

# TODO: rework it with matrix ops
get_box1_DefiniteRange = lambda lb, ub: logical_and(np.all(lb >= -1.0), np.all(ub <= 1.0))

st_arcsin = (lambda x: \
distribution.stochasticDistribution(arcsin(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([arcsin(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.arcsin(x))\
if hasStochastic\
else np.arcsin

def arcsin(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([arcsin(elem) for elem in inp])
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(arcsin(inp.values), inp.probabilities.copy())._update(inp)       
    if not isinstance(inp, oofun): 
        return np.arcsin(inp)
    r = oofun(st_arcsin, inp, d = lambda x: Diag(1.0 / np.sqrt(1.0 - x**2)), vectorized = True)
    r.getDefiniteRange = get_box1_DefiniteRange
    F_l, F_u = np.arcsin((-1, 1))
    r._interval_ = lambda domain, dtype: box_1_interval(inp, np.arcsin, domain, dtype, F_l, F_u)
    r.attach((inp>-1)('arcsin_domain_lower_bound_%d' % r._id, tol=-1e-7), (inp<1)('arcsin_domain_upper_bound_%d' % r._id, tol=-1e-7))
    return r

st_arccos = (lambda x: \
distribution.stochasticDistribution(arccos(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([arccos(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.arccos(x))\
if hasStochastic\
else np.arccos


def arccos(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([arccos(elem) for elem in inp])
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(arccos(inp.values), inp.probabilities.copy())._update(inp)     
    if not isinstance(inp, oofun): return np.arccos(inp)
    r = oofun(st_arccos, inp, d = lambda x: Diag(-1.0 / np.sqrt(1.0 - x**2)), vectorized = True)
    r.getDefiniteRange = get_box1_DefiniteRange
    F_l, F_u = np.arccos((-1, 1))
    r._interval_ = lambda domain, dtype: box_1_interval(inp, np.arccos, domain, dtype, F_l, F_u)
    r.attach((inp>-1)('arccos_domain_lower_bound_%d' % r._id, tol=-1e-7), (inp<1)('arccos_domain_upper_bound_%d' % r._id, tol=-1e-7))
    return r

st_arctan = (lambda x: \
distribution.stochasticDistribution(arctan(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([arctan(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.arctan(x))\
if hasStochastic\
else np.arctan

def arctan(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([arctan(elem) for elem in inp])    
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(arctan(inp.values), inp.probabilities.copy())._update(inp)             
    if not isinstance(inp, oofun): return np.arctan(inp)
    return oofun(st_arctan, inp, 
                 d = lambda x: Diag(1.0 / (1.0 + x**2)), 
                 vectorized = True, 
                 criticalPoints = False)

__all__ += ['arcsin', 'arccos', 'arctan']

st_sinh = (lambda x: \
distribution.stochasticDistribution(sinh(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([sinh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.sinh(x))\
if hasStochastic\
else np.sinh

def sinh(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([sinh(elem) for elem in inp])        
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(sinh(inp.values), inp.probabilities.copy())._update(inp)        
    if not isinstance(inp, oofun): return np.sinh(inp)
    return oofun(st_sinh, inp, d = lambda x: Diag(np.cosh(x)), vectorized = True, criticalPoints = False)


#def asdf(x):
##    print (1, type(x))
##    if isinstance(x, np.ndarray):
##        print('1>', x.shape)
##        print(x)
##        if isinstance(x[0], np.ndarray):
##            print(2, type(x[0]), x[0].shape)
##    if isinstance(x, multiarray):
##        print('-----')
##        print (x.shape, x.view(np.ndarray).shape)
##        print([type(elem) for elem in x.flat])
##        print '===='
#    r = distribution.stochasticDistribution(cosh(x.values), x.probabilities.copy())._update(x) \
#        if isinstance(x, distribution.stochasticDistribution)\
#        else np.array([cosh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray)\
#        else np.cosh(x)
#    return r
#
#st_cosh = (asdf)\
#if hasStochastic\
#else np.cosh


st_cosh = \
(lambda x: \
distribution.stochasticDistribution(cosh(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([cosh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.cosh(x))\
if hasStochastic\
else np.cosh

def cosh(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([cosh(elem) for elem in inp])        
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(cosh(inp.values), inp.probabilities.copy())._update(inp)                
    if not isinstance(inp, oofun): 
        return np.cosh(inp)
    return oofun(st_cosh, inp, d = lambda x: Diag(np.sinh(x)), vectorized = True, _interval_=ZeroCriticalPointsInterval(inp, np.cosh))
    
__all__ += ['sinh', 'cosh']

st_tanh = (lambda x: \
distribution.stochasticDistribution(tanh(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([tanh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.tanh(x))\
if hasStochastic\
else np.tanh


def tanh(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([tanh(elem) for elem in inp])       
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(tanh(inp.values), inp.probabilities.copy())._update(inp)              
    if not isinstance(inp, oofun): return np.tanh(inp)
    return oofun(st_tanh, inp, d = lambda x: Diag(1.0/np.cosh(x)**2), vectorized = True, criticalPoints = False)
    
st_arctanh = (lambda x: \
distribution.stochasticDistribution(arctanh(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([arctanh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.arctanh(x))\
if hasStochastic\
else np.arctanh

def arctanh(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([arctanh(elem) for elem in inp])        
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(arctanh(inp.values), inp.probabilities.copy())._update(inp)          
    if not isinstance(inp, oofun): return np.arctanh(inp)
    r = oofun(st_arctanh, inp, d = lambda x: Diag(1.0/(1.0-x**2)), vectorized = True, criticalPoints = False)
    r.getDefiniteRange = get_box1_DefiniteRange
    r._interval_ = lambda domain, dtype: box_1_interval(inp, np.arctanh, domain, dtype, -np.inf, np.inf)
    return r

__all__ += ['tanh', 'arctanh']

st_arcsinh = (lambda x: \
distribution.stochasticDistribution(arcsinh(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([arcsinh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.arcsinh(x))\
if hasStochastic\
else np.arcsinh

def arcsinh(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([arcsinh(elem) for elem in inp])        
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(arcsinh(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): return np.arcsinh(inp)
    return oofun(st_arcsinh, inp, d = lambda x: Diag(1.0/np.sqrt(1+x**2)), vectorized = True, criticalPoints = False)

st_arccosh = (lambda x: \
distribution.stochasticDistribution(arccosh(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([arccosh(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.arccosh(x))\
if hasStochastic\
else np.arccosh

def arccosh(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([arccosh(elem) for elem in inp])        
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(arccosh(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): return np.arccosh(inp)
    r = oofun(st_arccosh, inp, d = lambda x: Diag(1.0/np.sqrt(x**2-1.0)), vectorized = True)
    F0, shift = 0.0, 1.0
    r._interval_ = lambda domain, dtype: nonnegative_interval(inp, np.arccosh, domain, dtype, F0, shift)
    return r

__all__ += ['arcsinh', 'arccosh']

def angle(inp1, inp2):
    # returns angle between 2 vectors
    # TODO: 
    # 1) handle zero vector(s)
    # 2) handle small numerical errors more properly
    #     (currently they are handled via constraint attached to arccos)
    return arccos(sum(inp1*inp2)/sqrt(sum(inp1**2)*sum(inp2**2)))

st_exp = (lambda x: \
distribution.stochasticDistribution(exp(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([exp(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.exp(x))\
if hasStochastic\
else np.exp

def exp(inp):
    if isinstance(inp, ooarray):
        return ooarray([exp(elem) for elem in inp])         
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(exp(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): return np.exp(inp)
    return oofun(st_exp, inp, d = lambda x: Diag(np.exp(x)), vectorized = True, criticalPoints = False)

st_sqrt = (lambda x: \
distribution.stochasticDistribution(sqrt(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([sqrt(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.sqrt(x))\
if hasStochastic\
else np.sqrt

def sqrt(inp, attachConstraints = True):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([sqrt(elem) for elem in inp])
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(sqrt(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): 
        return np.sqrt(inp)
#    def fff(x):
#        print x
#        return np.sqrt(x)
    r = oofun(st_sqrt, inp, d = lambda x: Diag(0.5 / np.sqrt(x)), vectorized = True)
    F0 = 0.0
    r._interval_ = lambda domain, dtype: nonnegative_interval(inp, np.sqrt, domain, dtype, F0)
    if attachConstraints: r.attach((inp>0)('sqrt_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

__all__ += ['angle', 'exp', 'sqrt']

st_abs = (lambda x: \
distribution.stochasticDistribution(abs(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([abs(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.abs(x))\
if hasStochastic\
else np.abs

def abs(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([abs(elem) for elem in inp])
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(abs(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): return np.abs(inp)
    
    return oofun(st_abs, inp, d = lambda x: Diag(np.sign(x)), vectorized = True, _interval_ = ZeroCriticalPointsInterval(inp, np.abs))
    #return oofun(np.abs, inp, d = lambda x: Diag(np.sign(x)), vectorized = True, criticalPoints = ZeroCriticalPoints)

__all__ += ['abs']

def log_interval(logfunc, inp):
    def interval(domain, dtype):
        lb_ub, definiteRange = inp._interval(domain, dtype)
        lb, ub = lb_ub[0], lb_ub[1]
        t_min, t_max = atleast_1d(logfunc(lb)), atleast_1d(logfunc(ub))
        
        ind = lb < 0
        if np.any(ind):
            ind2 = ub>0
            t_min[atleast_1d(logical_and(ind, ind2))] = -np.inf
            definiteRange = False
        ind = atleast_1d(ub == 0)
        if np.any(ind):
            t_max[ind] = np.nan
            t_min[ind] = np.nan
            definiteRange = False
        #print definiteRange
        # TODO: rework definiteRange with matrix operations
        
        return np.vstack((t_min, t_max)), definiteRange
    return interval

st_log = (lambda x: \
distribution.stochasticDistribution(log(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([log(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.log(x))\
if hasStochastic\
else np.log

def log(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([log(elem) for elem in inp])    
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(log(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): return np.log(inp)
    r = oofun(st_log, inp, d = lambda x: Diag(1.0/x), vectorized = True, _interval_ = log_interval(np.log, inp))
    r.attach((inp>1e-300)('log_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

st_log10 = (lambda x: \
distribution.stochasticDistribution(log10(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([log10(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.log10(x))\
if hasStochastic\
else np.log10

INV_LOG_10 = 1.0 / np.log(10)
def log10(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([log10(elem) for elem in inp])    
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(log10(inp.values), inp.probabilities.copy())._update(inp)              
    if not isinstance(inp, oofun): return np.log10(inp)
    r = oofun(st_log10, inp, d = lambda x: Diag(INV_LOG_10 / x), vectorized = True, _interval_ = log_interval(np.log10, inp))
    r.attach((inp>1e-300)('log10_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

st_log2 = (lambda x: \
distribution.stochasticDistribution(log2(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([log2(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.log2(x))\
if hasStochastic\
else np.log2

INV_LOG_2 = 1.0 / np.log(2)
def log2(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([log2(elem) for elem in inp])    
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(log2(inp.values), inp.probabilities.copy())._update(inp)       
    if not isinstance(inp, oofun): return np.log2(inp)
    r = oofun(st_log2, inp, d = lambda x: Diag(INV_LOG_2/x), vectorized = True, _interval_ = log_interval(np.log2, inp))
    r.attach((inp>1e-300)('log2_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

__all__ += ['log', 'log2', 'log10']

def dot(inp1, inp2):
    if not isinstance(inp1, oofun) and not isinstance(inp2, oofun): return np.dot(inp1, inp2)
    
    def aux_d(x, y):
        if y.size == 1: 
            r = np.empty_like(x)
            r.fill(y)
            return Diag(r)
        else:
            return y
        
    r = oofun(lambda x, y: x * y if x.size == 1 or y.size == 1 else np.dot(x, y), [inp1, inp2], d=(lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x)))
    r.getOrder = lambda *args, **kwargs: (inp1.getOrder(*args, **kwargs) if isinstance(inp1, oofun) else 0) + (inp2.getOrder(*args, **kwargs) if isinstance(inp2, oofun) else 0)
    #r.isCostly = True
    return r

def cross(a, b):
    if not isinstance(a, oofun) and not isinstance(b, oofun): return np.cross(a, b)

    
    def aux_d(x, y):
        assert x.size == 3 and y.size == 3, 'currently FuncDesigner cross(x,y) is implemented for arrays of length 3 only'
        return np.array([[0, -y[2], y[1]], [y[2], 0, -y[0]], [-y[1], y[0], 0]])
   
    r = oofun(lambda x, y: np.cross(x, y), [a, b], d=(lambda x, y: -aux_d(x, y), lambda x, y: aux_d(y, x)))
    r.getOrder = lambda *args, **kwargs: (a.getOrder(*args, **kwargs) if isinstance(a, oofun) else 0) + (b.getOrder(*args, **kwargs) if isinstance(b, oofun) else 0)
    return r

__all__ += ['dot', 'cross']

def ceil(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([ceil(elem) for elem in inp])        
    if not isinstance(inp, oofun): return np.ceil(inp)
    r = oofun(lambda x: np.ceil(x), inp, vectorized = True)
    r._D = lambda *args, **kwargs: raise_except('derivative for FD ceil is unimplemented yet')
    r.criticalPoints = False#lambda arg_infinum, arg_supremum: [np.ceil(arg_supremum)]
    return r

def floor(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([floor(elem) for elem in inp])        
    if not isinstance(inp, oofun): return np.floor(inp)
    r = oofun(lambda x: np.floor(x), inp, vectorized = True)
    r._D = lambda *args, **kwargs: raise_except('derivative for FD floor is unimplemented yet')
    r.criticalPoints = False#lambda arg_infinum, arg_supremum: [np.floor(arg_infinum)]
    return r

st_sign = (lambda x: \
distribution.stochasticDistribution(sign(x.values), x.probabilities.copy())._update(x) \
if isinstance(x, distribution.stochasticDistribution)\
else np.array([sign(elem) for elem in x.flat]).view(multiarray) if isinstance(x, multiarray) and isinstance(x.flat[0], distribution.stochasticDistribution)
else np.sign(x))\
if hasStochastic\
else np.sign

def sign(inp):
    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]):
        return ooarray([sign(elem) for elem in inp])
    if hasStochastic and  isinstance(inp, distribution.stochasticDistribution):
        return distribution.stochasticDistribution(sign(inp.values), inp.probabilities.copy())._update(inp)      
    if not isinstance(inp, oofun): 
        return np.sign(inp)
    r = oofun(st_sign, inp, vectorized = True, d = lambda x: 0.0)
    r.criticalPoints = False
    return r

__all__ += ['ceil', 'floor', 'sign']


def sum_engine(r0, *args):
    if not hasStochastic:
        return PythonSum(args) + r0
    Args, Args_st = [], {}
    for elem in args:
        if isinstance(elem, distribution.stochasticDistribution):
            stDep = frozenset(elem.stochDep.keys())
            tmp = Args_st.get(stDep, None)
            if tmp is None:
                Args_st[stDep] = [elem]
            else:
                Args_st[stDep].append(elem)
        else:
            Args.append(elem)
    r = PythonSum(Args) + r0
    if len(Args_st) == 0:
        return r
    
    # temporary
    for key, val in Args_st.items():
        maxDistributionSize = val[0].maxDistributionSize
        break
    stValues = Args_st.values()
#            stValues = Args_st.values()
#            T = list(set(stValues))[0]
#            maxDistributionSize = next(iter(T)).maxDistributionSize
    r1 = 0.0
    for elem in stValues:
        tmp = PythonSum(elem)
        r1 = tmp + r1 
        r1.reduce(maxDistributionSize)
    r1 = r1 + r
    r1.maxDistributionSize = maxDistributionSize
    return r1 

def sum_interval(R0, r, INP, domain, dtype):
   
    v = domain.modificationVar
    if v is not None:
        # self already must be in domain.storedSums
        R, DefiniteRange = domain.storedSums[r][-1]
        if not np.all(np.isfinite(R)):
            R = np.asarray(R0, dtype).copy()
            if domain.isMultiPoint:
                R = np.tile(R, (1, len(list(domain.values())[0][0])))
                #R = np.tile(R, (1, len(domain.values()[0][0])))
            DefiniteRange = True
            #####################
            # !!! don't use sum([inp._interval(domain, dtype) for ...]) here
            # to reduce memory consumption
            for inp in INP:
                arg_lb_ub, definiteRange = inp._interval(domain, dtype)
#                DefiniteRange = logical_and(DefiniteRange, definiteRange)
                if R.shape == arg_lb_ub.shape:
                    R += arg_lb_ub
                else:
                    R = R + arg_lb_ub
            #####################
            return R, DefiniteRange

        R=R.copy()
        for inp in r.storedSumsFuncs[v]:
            # TODO: mb rework definiteRange processing ?
            arg_lb_ub, definiteRange = inp._interval(domain, dtype)
            DefiniteRange = logical_and(DefiniteRange, definiteRange)
            R += arg_lb_ub

        R -= domain.storedSums[r][v]
        
        # To supress inf-inf=nan, however, it doesn't work properly yet, other code is used
        if np.any(np.isinf(arg_lb_ub)):
            R[arg_lb_ub == np.inf] = np.inf
            R[arg_lb_ub == -np.inf] = -np.inf
        
        return R, DefiniteRange
    else:
        domain.storedSums[r] = {}        

    #assert np.asarray(r0).ndim <= 1
    #R = np.asarray(R0, dtype).copy()
    R = np.asarray(R0).copy()
    if domain.isMultiPoint:
        R = np.tile(R, (1, len(list(domain.values())[0][0])))

    #####################
    # !!! don't use sum([inp._interval(domain, dtype) for ...]) here
    # to reduce memory consumption
    DefiniteRange = True
#            R_ = []
    D = domain.storedSums[r]
    for inp in INP:
        arg_lb_ub, definiteRange = inp._interval(domain, dtype)
        Tmp = inp._getDep() if not inp.is_oovar else [inp]
        for oov in Tmp:
            tmp = D.get(oov, None)
            if tmp is None:
                D[oov] = arg_lb_ub.copy()
            else:
                try:
                    D[oov] += arg_lb_ub
                except:
                    # may be of different shape, e.g. for a fixed variable
                    D[oov] = D[oov] + arg_lb_ub
        
        DefiniteRange = logical_and(DefiniteRange, definiteRange)
        if R.shape == arg_lb_ub.shape:
            R += arg_lb_ub
        else:
            R = R + arg_lb_ub
    #####################
    
    if v is None:
        domain.storedSums[r][-1] = R, DefiniteRange
        
    return R, DefiniteRange


def sum_derivative(r_, r0, INP, dep, point, fixedVarsScheduleID, Vars=None, fixedVars = None, useSparse = 'auto'):
    # TODO: handle involvePrevData
    # TODO: handle fixed vars
    
    r = {}
   
    isSP = hasattr(point, 'maxDistributionSize') and point.maxDistributionSize != 0
    
    for elem in INP:
        if not elem.is_oovar and (elem.input is None or len(elem.input)==0 or elem.input[0] is None): 
            continue # TODO: get rid if None, use [] instead
        if elem.discrete: continue
        
        # TODO: code cleanup 
        if elem.is_oovar:
            if (fixedVars is not None and elem in fixedVars) or (Vars is not None and elem not in Vars): continue
            sz = np.asarray(point[elem]).size
            tmpres = Eye(sz) if not isinstance(point[elem], multiarray) else np.ones(sz).view(multiarray)
            r_val = r.get(elem, None)
            if isSP:
                if r_val is not None:
                    r_val.append(tmpres)
                else:
                    r[elem] = [tmpres]
            else:
                if r_val is not None:
                    if sz != 1 and isinstance(r_val, np.ndarray) and not isinstance(tmpres, np.ndarray): # i.e. tmpres is sparse matrix
                        tmpres = tmpres.toarray()
                    elif not np.isscalar(r_val) and not isinstance(r_val, np.ndarray) and isinstance(tmpres, np.ndarray):
                        r[elem] = r_val.toarray()
                    Tmp = tmpres.resolve(True) if isspmatrix(r[elem]) and type(tmpres) == DiagonalType else tmpres
                    try:
                        r[elem] += Tmp
                    except:
                        r[elem] = r[elem] + Tmp
                else:
                    # TODO: check it for oovars with size > 1
                    r[elem] = tmpres
        else:
            tmp = elem._D(point, fixedVarsScheduleID, Vars, fixedVars, useSparse = useSparse)
            for key, val in tmp.items():
                r_val = r.get(key, None)
                if isSP:
                    if r_val is not None:
                        r_val.append(val)
                    else:
                        r[key] = [val]
                else:
                    if r_val is not None:
                        if not np.isscalar(val) and isinstance(r_val, np.ndarray) and not isinstance(val, np.ndarray): # i.e. tmpres is sparse matrix
                            val = val.toarray()
                        elif not np.isscalar(r_val) and not isinstance(r_val, np.ndarray) and isinstance(val, np.ndarray):
                            r[key] = r_val.toarray()
                        
                        if isspmatrix(r_val) and type(val) == DiagonalType:
                            val = val.resolve(True)
                        elif isspmatrix(val) and type(r_val) == DiagonalType:
                            r[key] = r_val.resolve(True)
                        
                        # TODO: rework it
                        try:
                            r[key] += val
                        except:
                            r[key] = r_val + val
                    else:
                        r[key] = Copy(val)
    
    if isSP:
        for key, val in r.items():
            r[key] = sum_engine(0.0, *val)
            
    
    if useSparse is False:
        for key, val in r.items():
            #if np.isscalar(val): val = np.asfarray(val)
            if hasattr(val, 'toarray'):# and not isinstance(val, multiarray): # i.e. sparse matrix
                r[key] = val.toarray()

    if not isSP:
        # TODO: rework it, don't recalculate each time
        Size = np.asarray(r0).size
        if Size == 1 and not point.isMultiPoint:
            if r_._lastFuncVarsID == fixedVarsScheduleID:
                if not np.isscalar(r_._f_val_prev):
                    Size = r_._f_val_prev.size
            else:
                Size = np.asarray(r_._getFuncCalcEngine(point, Vars = Vars, fixedVars = fixedVars, fixedVarsScheduleID = fixedVarsScheduleID)).size

        if Size != 1 and not point.isMultiPoint:
            for key, val in r.items():
                if not isinstance(val, diagonal):
                    if np.isscalar(val) or np.prod(val.shape) <= 1:
                        tmp = np.empty((Size, 1))
                        tmp.fill(val if np.isscalar(val) else val.item())
                        r[key] = tmp
                    elif val.shape[0] != Size:
                        tmp = np.tile(val, (Size, 1))
                        r[key] = tmp
    #                    elif np.asarray(val).size !=1:
    #                        raise_except('incorrect size in FD sum kernel')
    
    return r

def sum_getOrder(INP, *args, **kwargs):
    orders = [0]+[inp.getOrder(*args, **kwargs) for inp in INP]
    return np.max(orders)


def sum(inp, *args, **kwargs):
    if type(inp) == np.ndarray and inp.dtype != object:
        return np.sum(inp, *args, **kwargs)
        
    if isinstance(inp, ooarray) and inp.dtype != object:
        inp = inp.view(np.ndarray)
        
    cond_ooarray = isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)])
    if cond_ooarray and inp.size == 1: 
        return np.asscalar(inp).sum()
    condIterableOfOOFuns = type(inp) in (list, tuple) or cond_ooarray
    
    if not isinstance(inp, oofun) and not condIterableOfOOFuns: 
        return np.sum(inp, *args, **kwargs)

    if isinstance(inp, ooarray) and any([isinstance(elem, oofun) for elem in atleast_1d(inp)]): inp = inp.tolist()

    if condIterableOfOOFuns:
        d, INP, r0 = [], [], 0.0
        for elem in inp: # TODO: mb use reduce() or something like that
            if not isinstance(elem, oofun): 
                # not '+=' because size can be changed from 1 to another value
                r0 = r0 + np.asanyarray(elem) # so it doesn't work for different sizes
                continue
            INP.append(elem)
        if len(INP) == 0:
            return r0
        
        r = oofun(lambda *args: sum_engine(r0, *args), INP, _isSum = True)
        r._summation_elements = INP if np.isscalar(r0) and r0 == 0.0 else INP + [r0]

        r.storedSumsFuncs = {}
        for inp in INP:
            Dep = [inp] if inp.is_oovar else inp._getDep()
            for v in Dep:
                if v not in r.storedSumsFuncs:
                    r.storedSumsFuncs[v] = set()
                r.storedSumsFuncs[v].add(inp)
                                
        # TODO:  check for fixed inputs
        
        r.getOrder = lambda *args, **kw: sum_getOrder(INP, *args, **kw)
        
        R0 = np.tile(r0, (2, 1))

        r._interval_ = lambda *args, **kw: sum_interval(R0, r, INP, *args, **kw)
        r.vectorized = True
        r_dep = r._getDep()
        r._D = lambda *args, **kw: sum_derivative(r, r0, INP, r_dep, *args, **kw)
#        r.isCostly = True
        return r
    else: 
        return inp.sum(*args, **kwargs)#np.sum(inp, *args, **kwargs)
    
    
def prod(inp, *args, **kwargs):
    if not isinstance(inp, oofun): return np.prod(inp, *args, **kwargs)
    if len(args) != 0 or len(kwargs) != 0:
        raise FuncDesignerException('oofun for prod(x, *args,**kwargs) is not implemented yet')
    #r.getOrder = lambda *args, **kwargs: prod([(1 if not isinstance(inp, oofun) else inp.getOrder(*args, **kwargs)) for inp in self.input])
    return inp.prod()

# Todo: implement norm_1, norm_inf etc
def norm(*args, **kwargs):
    if len(kwargs) or len(args) > 1:
        return np.linalg.norm(*args, **kwargs)
    r = sqrt(sum(args[0]**2),  attachConstraints=False)
    if isinstance(r, oofun):
        r.hasDefiniteRange=True
    return r

__all__ += ['sum', 'prod', 'norm']

#def stack(*args, **kwargs):
#    assert len(kwargs) == 0 and len(args) != 0
#    if len(args) == 1:
#        assert type(args[0]) in (list, tuple)
#        if not any([isinstance(arg, oofun) for arg in args[0]]): return np.hstack(args)
#        #f = lambda *Args: np.hstack([arg(Args) if isinstance(arg, oofun) else arg for arg in args[0]])
#        def f(*Args): 
#            r = np.hstack([arg.fun(Args) if isinstance(arg, oofun) else arg for arg in args[0]])
#            print '1:', r
#            raise 0
#            return r
#        #d = lambda *Args: np.hstack([arg.d(Args).reshape(-1, 1) if isinstance(arg, oofun) else np.zeros((len(arg))) for arg in args[0]])
#        def d(*Args):
#            r = np.hstack([arg.d(Args).reshape(-1, 1) if isinstance(arg, oofun) else np.zeros((len(arg))) for arg in args[0]])
#            print '2:', r
#            return r
#        print 'asdf', args[0]
#        return oofun(f, args[0], d=d)
#    else:
#        raise FuncDesignerException('unimplemented yet')
#        #assert isinstance(args[0], oofun) 
        
    

#def norm(inp, *args, **kwargs):
#    if len(args) != 0 or len(kwargs) != 0:
#        raise FuncDesignerException('oofun for norm(x, *args,**kwargs) is not implemented yet')
#    
#    #np.linalg.norm
#    f = lambda x: np.sqrt(np.sum(x**2))
#    
#    r = oofun(f, inp, isCostly = True)
#    
#    def d(x):
#        
#    
#    #r.d = lambda *args, **kwargs: 
#        
#        #s = r(x)
#        #return Diag(x /  s if s != 0 else np.zeros(x.size)) # however, dirivative doesn't exist in (0,0,..., 0)
#    r.d = d
#    
#    return r
    

def size(inp, *args, **kwargs):
    if not isinstance(inp, oofun): return np.size(inp, *args, **kwargs)
    return inp.size
    
def ifThenElse(condition, val1, val2, *args, **kwargs):
    
    # for future implementation
    assert len(args) == 0 and len(kwargs) == 0 
    Val1 = atleast_oofun(val1)#fixed_oofun(val1) if not isinstance(val1, oofun) else val1
    #if np.isscalar(val1): raise 0
    Val2 = atleast_oofun(val2)#fixed_oofun(val2) if not isinstance(val2, oofun) else val2
    if isinstance(condition, bool): 
        return Val1 if condition else Val2
    elif isinstance(condition, oofun):
        f = lambda conditionResult, value1Result, value2Result: value1Result if conditionResult else value2Result
        # !!! Don't modify it elseware function will evaluate both expressions despite of condition value 
        r = oofun(f, [condition, val1, val2])
        r.D = lambda point, *args, **kwargs: (Val1.D(point, *args, **kwargs) if isinstance(Val1, oofun) else {}) if condition(point) else \
        (Val2.D(point, *args, **kwargs) if isinstance(Val2, oofun) else {})
        r._D = lambda point, *args, **kwargs: (Val1._D(point, *args, **kwargs) if isinstance(Val1, oofun) else {}) if condition(point) else \
        (Val2._D(point, *args, **kwargs) if isinstance(Val2, oofun) else {})
        r.d = errFunc
        
        # TODO: try to set correct value from val1, val2 if condition is fixed
#        def getOrder(Vars=None, fixedVars=None, *args, **kwargs):
#            dep = condition.getDep()
#            if Vars is not None and dep.is

        return r
    else:
        raise FuncDesignerException('ifThenElse requires 1st argument (condition) to be either boolean or oofun, got %s instead' % type(condition))

__all__ += ['size', 'ifThenElse']

def decision(*args, **kwargs):
    pass
        
def max(inp,  *args,  **kwargs): 
    if type(inp) in (list, tuple, np.ndarray) \
    and (len(args) == 0 or len(args) == 1 and not isinstance(args[0], oofun)) \
    and not any([isinstance(elem, oofun) for elem in (inp if type(inp) in (list, tuple) else np.atleast_1d(inp))]):
        return np.max(inp, *args, **kwargs)
        
    assert len(args) == len(kwargs) == 0, 'incorrect data type in FuncDesigner max or not implemented yet'
    
    if isinstance(inp, oofun):
        f = lambda x: np.max(x)
#        def f(x):
#            print np.max(x)
#            return np.max(x)
        def d(x):
            df = inp.d(x)
            ind = np.argmax(x)
            return df[ind, :]
        def interval(domain, dtype):
            lb_ub, definiteRange = inp._interval(domain, dtype)
            tmp1, tmp2 = lb_ub[0], lb_ub[1]
            return np.vstack((np.max(np.vstack(tmp1), 0), np.max(np.vstack(tmp2), 0))), np.all(definiteRange, 0)
        r = oofun(f, inp, d = d, size = 1, _interval_ = interval)
    elif type(inp) in (list, tuple, ooarray):
        f = lambda *args: np.max([arg for arg in args])
        def interval(domain, dtype):
            arg_inf, arg_sup, tmp, DefiniteRange = [], [], -np.inf, True
            for _inp in inp:
                if isinstance(_inp, oofun):
                    #tmp1, tmp2 = _inp._interval(domain, dtype)
                    lb_ub, definiteRange = _inp._interval(domain, dtype)
                    tmp1, tmp2 = lb_ub[0], lb_ub[1]
                    arg_inf.append(tmp1)
                    arg_sup.append(tmp2)
                    DefiniteRange = logical_and(DefiniteRange, definiteRange)
                elif tmp < _inp:
                    tmp = _inp
            r1, r2 = np.max(np.vstack(arg_inf), 0), np.max(np.vstack(arg_sup), 0)
            r1[r1<tmp] = tmp
            r2[r2<tmp] = tmp
            return np.vstack((r1, r2)), DefiniteRange
        r = oofun(f, inp, size = 1, _interval_ = interval)
        def _D(point, *args, **kwargs):
            ind = np.argmax([(s(point) if isinstance(s, oofun) else s) for s in r.input])
            return r.input[ind]._D(point, *args, **kwargs) if isinstance(r.input[ind], oofun) else {}
        r._D = _D
    else:
        return np.max(inp, *args, **kwargs)
    return r        
    
def min(inp,  *args,  **kwargs): 
    if type(inp) in (list, tuple, np.ndarray) \
    and (len(args) == 0 or len(args) == 1 and not isinstance(args[0], oofun))\
    and not any([isinstance(elem, oofun) for elem in (inp if type(inp) in (list, tuple) else np.atleast_1d(inp))]):
        return np.min(inp, *args, **kwargs)
    
    assert len(args) == len(kwargs) == 0, 'incorrect data type in FuncDesigner min or not implemented yet'
    if isinstance(inp, oofun):
        f = lambda x: np.min(x)
        def d(x):
            df = inp.d(x)
            #df = inp.d(x) if type(inp.d) not in (list, tuple) else np.hstack([item(x) for item in inp.d])
            ind = np.argmin(x)
            return df[ind, :]
        def interval(domain, dtype):
            lb_ub, definiteRange = inp._interval(domain, dtype)
            tmp1, tmp2 = lb_ub[0], lb_ub[1]
            return np.vstack((np.min(np.vstack(tmp1), 0), np.min(np.vstack(tmp2), 0))), np.all(definiteRange, 0)
        r = oofun(f, inp, d = d, size = 1, _interval_ = interval)
    elif type(inp) in (list, tuple, ooarray):
        f = lambda *args: np.min([arg for arg in args])
        def interval(domain, dtype):
            arg_inf, arg_sup, tmp, DefiniteRange = [], [], np.inf, True
            for _inp in inp:
                if isinstance(_inp, oofun):
                    lb_ub, definiteRange = _inp._interval(domain, dtype)
                    tmp1, tmp2 = lb_ub[0], lb_ub[1]
                    arg_inf.append(tmp1)
                    arg_sup.append(tmp2)
                    DefiniteRange = logical_and(DefiniteRange, definiteRange)
                elif tmp > _inp:
                    tmp = _inp
            r1, r2 = np.min(np.vstack(arg_inf), 0), np.min(np.vstack(arg_sup), 0)
            if np.isfinite(tmp):
                r1[r1>tmp] = tmp
                r2[r2>tmp] = tmp
            return np.vstack((r1, r2)), DefiniteRange
            
        r = oofun(f, inp, size = 1, _interval_ = interval)
        def _D(point, *args, **kwargs):
            ind = np.argmin([(s(point) if isinstance(s, oofun) else s) for s in r.input])
            return r.input[ind]._D(point, *args, **kwargs) if isinstance(r.input[ind], oofun) else {}
        r._D = _D
    else:
        return np.min(inp, *args, **kwargs)
    return r        


__all__ += ['min', 'max']

#def fixed_oofun(Val):
#    val = np.asfarray(Val)
#    f = lambda: Val
#    r = oofun(f, input=[])
#    r._D = lambda *args,  **kwargs: {}
#    r.D = lambda *args,  **kwargs: {}
#    r.discrete = True
#    return r

det3 = lambda a, b, c: a[0] * (b[1]*c[2] - b[2]*c[1]) - a[1] * (b[0]*c[2] - b[2]*c[0]) + a[2] * (b[0]*c[1] - b[1]*c[0]) 

__all__ += ['det3']


def hstack(tup): # overload for oofun[ind]
    c = [isinstance(t, (oofun, ooarray)) for t in tup]
    if any([isinstance(t, ooarray) for t in tup]):
        return ooarray(np.hstack(tup))
    if not any(c):
        return np.hstack(tup)
    #an_oofun_ind = np.where(c)[0][0]
    f = lambda *x: np.hstack(x)
    
    
  
#    def d(*x): 
#
#        r = [elem.d(x[i]) if c[i] else None for i, elem in enumerate(tup)]
#        size = atleast_1d(r[an_oofun_ind]).shape[0]
#        r2 = [elem if c[i] else Zeros(size) for elem in r]
#        return r2
        
        #= lambda *x: np.hstack([elem.d(x) if c[i] else elem for elem in tup])
            
#        f = lambda x: x[ind] 
#        def d(x):
#            Xsize = Len(x)
#            condBigMatrix = Xsize > 100 
#            if condBigMatrix and scipyInstalled:
#                r = SparseMatrixConstructor((1, x.shape[0]))
#                r[0, ind] = 1.0
#            else: 
#                if condBigMatrix and not scipyInstalled: self.pWarn(scipyAbsentMsg)
#                r = zeros_like(x)
#                r[ind] = 1
#            return r
    def getOrder(*args, **kwargs):
        orders = [0]+[inp.getOrder(*args, **kwargs) for inp in tup]
        return np.max(orders)
    
            
    r = oofun(f, tup, getOrder = getOrder)
    
    #!!!!!!!!!!!!!!!!! TODO: sparse 

    
    def _D(*args,  **kwargs): 
        # TODO: rework it, especially if sizes are fixed and known
        # TODO: get rid of fixedVarsScheduleID
        sizes = [(t(args[0], fixedVarsScheduleID = kwargs.get('fixedVarsScheduleID', -1)) if c[i] else np.asarray(t)).size for i, t in enumerate(tup)]
        
        tmp = [elem._D(*args,  **kwargs) if c[i] else None for i, elem in enumerate(tup)]
        res = {}
        for v in r._getDep():
            Temp = []
            for i, t in enumerate(tup):
                if c[i]:
                    temp = tmp[i].get(v, None)
                    if temp is not None:
                        Temp.append(temp if type(temp) != DiagonalType else temp.resolve(kwargs['useSparse']))
                    else:
#                        T = next(iter(tmp[i].values()))
#                        sz = T.shape[0] if type(T) == DiagonalType else np.atleast_1d(T).shape[0]
                        Temp.append((Zeros if sizes[i] * np.asarray(args[0][v]).size > 1000 else np.zeros)((sizes[i], np.asarray(args[0][v]).size)))
                else:
                    sz = np.atleast_1d(t).shape[0]
                    Temp.append(Zeros((sz, 1)) if sz > 100 else np.zeros(sz))
            rr = Vstack([elem for elem in Temp])
            #print type(rr)
            res[v] = rr if not isspmatrix(rr) or 0.3 * prod(rr.shape) > rr.size else rr.toarray()
            #print type(res[v])
        return res
    r._D = _D
    return r

__all__ += ['hstack']

# TODO: move the func into fdmisc.py
def errFunc(*args,  **kwargs): 
    # this function shouldn't be ever called, an FD kernel hack has been involved
    raise FuncDesignerException('error in FuncDesigner kernel, inform developers')

#for func in (sin, arctan):
#    i0 = func._interval
#    def f2(domain, dtype):
#        if type(domain) == dict:
#            return i0(domain, dtype)
#        r = domain.storedIntervals.get(self, None)
#        if r is None:
#            r = i0(domain, dtype)
#            domain.storedIntervals[self] = r
#        return r
#    func._interval = f2
