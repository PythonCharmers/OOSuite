from ooFun import oofun
import numpy as np
from misc import FuncDesignerException, Diag, Eye, raise_except
from ooFun import atleast_oofun, ooarray
from ooPoint import ooPoint
from Interval import TrigonometryCriticalPoints, ZeroCriticalPoints, nonnegative_interval
from numpy import atleast_1d, logical_and, logical_not, empty_like

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



def sin(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([sin(elem) for elem in inp])
    elif not isinstance(inp, oofun): return np.sin(inp)
    return oofun(np.sin, inp, 
                 d = lambda x: Diag(np.cos(x)), 
                 vectorized = True, 
                 criticalPoints = TrigonometryCriticalPoints)
                 #_interval = lambda domain: ufuncInterval(inp, domain, np.sin, TrigonometryCriticalPoints))

def cos(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([cos(elem) for elem in inp])
    elif not isinstance(inp, oofun): return np.cos(inp)
    #return oofun(np.cos, inp, d = lambda x: Diag(-np.sin(x)))
    return oofun(np.cos, inp, 
             d = lambda x: Diag(-np.sin(x)), 
             vectorized = True, 
             criticalPoints = TrigonometryCriticalPoints)
             #_interval = lambda domain: ufuncInterval(inp, domain, np.cos, TrigonometryCriticalPoints))

def tan(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([tan(elem) for elem in inp])
    if not isinstance(inp, oofun): return np.tan(inp)
    # TODO: move it outside of tan definition
    def interval(arg_inf, arg_sup):
        raise 'interval for tan is unimplemented yet'
    r = oofun(np.tan, inp, d = lambda x: Diag(1.0 / np.cos(x) ** 2), vectorized = True, interval = interval)
    return r

# TODO: cotan?

def arcsin(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([arcsin(elem) for elem in inp])
    if not isinstance(inp, oofun): 
        return np.arcsin(inp)
    r = oofun(np.arcsin, inp, d = lambda x: Diag(1.0 / np.sqrt(1.0 - x**2)), vectorized = True)
    r.criticalPoints = False
    r.attach((inp>-1)('arcsin_domain_lower_bound_%d' % r._id, tol=-1e-7), (inp<1)('arcsin_domain_upper_bound_%d' % r._id, tol=-1e-7))
    return r

def arccos(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([arccos(elem) for elem in inp])
    if not isinstance(inp, oofun): return np.arccos(inp)
    r = oofun(np.arccos, inp, d = lambda x: Diag(-1.0 / np.sqrt(1.0 - x**2)), vectorized = True)
    r.criticalPoints = False
    r.attach((inp>-1)('arccos_domain_lower_bound_%d' % r._id, tol=-1e-7), (inp<1)('arccos_domain_upper_bound_%d' % r._id, tol=-1e-7))
    return r

def arctan(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([arctan(elem) for elem in inp])    
    if not isinstance(inp, oofun): return np.arctan(inp)
    return oofun(np.arctan, inp, 
                 d = lambda x: Diag(1.0 / (1.0 + x**2)), 
                 vectorized = True, 
                 criticalPoints = False)

def sinh(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([sinh(elem) for elem in inp])        
    if not isinstance(inp, oofun): return np.sinh(inp)
    return oofun(np.sinh, inp, d = lambda x: Diag(np.cosh(x)), vectorized = True, criticalPoints = False)

def cosh(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([cosh(elem) for elem in inp])        
    if not isinstance(inp, oofun): return np.cosh(inp)
    return oofun(np.cosh, inp, d = lambda x: Diag(np.sinh(x)), vectorized = True, criticalPoints=ZeroCriticalPoints)

def angle(inp1, inp2):
    # returns angle between 2 vectors
    # TODO: 
    # 1) handle zero vector(s)
    # 2) handle small numerical errors more properly
    #     (currently they are handled via constraint attached to arccos)
    return arccos(sum(inp1*inp2)/sqrt(sum(inp1**2)*sum(inp2**2)))

def exp(inp):
    if isinstance(inp, ooarray):
        return ooarray([exp(elem) for elem in inp])            
    if not isinstance(inp, oofun): return np.exp(inp)
    return oofun(np.exp, inp, d = lambda x: Diag(np.exp(x)), vectorized = True, criticalPoints = False)

       
def sqrt(inp, attachConstraints = True):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([sqrt(elem) for elem in inp])
    elif not isinstance(inp, oofun): 
        return np.sqrt(inp)
    r = oofun(np.sqrt, inp, d = lambda x: Diag(0.5 / np.sqrt(x)), vectorized = True)
    r._interval = lambda domain, dtype: nonnegative_interval(inp, np.sqrt, domain, dtype)
    if attachConstraints: r.attach((inp>0)('sqrt_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

def abs(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([abs(elem) for elem in inp])
    elif not isinstance(inp, oofun): return np.abs(inp)
    return oofun(np.abs, inp, d = lambda x: Diag(np.sign(x)), vectorized = True, criticalPoints = ZeroCriticalPoints)

def log_interval(logfunc, inp):
    def interval(domain, dtype):
        lb, ub = inp._interval(domain, dtype)
        
        ind = lb <=0
        if any(ind):
            t_min, t_max = atleast_1d(empty_like(lb)), atleast_1d(empty_like(ub))
            ind_dom = logical_not(ind)
            t_min[ind_dom], t_max[ind_dom] = logfunc(lb[ind_dom]), logfunc(ub[ind_dom])
            #t_min, t_max = np.asarray(t_min), np.asarray(t_max)
            ind2 = ub>0
            t_min[atleast_1d(logical_and(ind, ind2))] = -np.inf
            ind_nan = logical_and(ind, logical_not(ind2))
            t_min[atleast_1d(ind_nan)] = np.nan
            t_max[atleast_1d(ind_nan)] = np.nan
        else:
            t_min, t_max = logfunc(lb), logfunc(ub)
        return t_min, t_max
    return interval

def log(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([log(elem) for elem in inp])    
    if not isinstance(inp, oofun): return np.log(inp)
    r = oofun(np.log, inp, d = lambda x: Diag(1.0/x), vectorized = True, _interval = log_interval(np.log, inp))
    r.attach((inp>1e-300)('log_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

INV_LOG_10 = 1.0 / np.log(10)
def log10(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([log10(elem) for elem in inp])    
    if not isinstance(inp, oofun): return np.log10(inp)
    r = oofun(np.log10, inp, d = lambda x: Diag(INV_LOG_10 / x), vectorized = True, _interval = log_interval(np.log10, inp))
    r.attach((inp>1e-300)('log10_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r
    
INV_LOG_2 = 1.0 / np.log(2)
def log2(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([log2(elem) for elem in inp])    
    if not isinstance(inp, oofun): return np.log2(inp)
    r = oofun(np.log2, inp, d = lambda x: Diag(INV_LOG_2/x), vectorized = True, _interval = log_interval(np.log2, inp))
    r.attach((inp>1e-300)('log2_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

def dot(inp1, inp2):
    if not isinstance(inp1, oofun) and not isinstance(inp2, oofun): return np.dot(inp1, inp2)
    
    def aux_d(x, y):
        if y.size == 1: 
            #r = np.empty(x.size) - use it?
            r = np.empty_like(x)
            r.fill(y)
            r = Diag(r)
        else:
            r = np.copy(y)
        return r
        
    r = oofun(lambda x, y: x * y if x.size == 1 or y.size == 1 else np.dot(x, y), [inp1, inp2], d=(lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x)))
    r.getOrder = lambda *args, **kwargs: (inp1.getOrder(*args, **kwargs) if isinstance(inp1, oofun) else 0) + (inp2.getOrder(*args, **kwargs) if isinstance(inp2, oofun) else 0)
    r.isCostly = True
    return r

def cross(a, b):
    if not isinstance(a, oofun) and not isinstance(b, oofun): return np.cross(a, b)

    
    def aux_d(x, y):
        assert x.size == 3 and y.size == 3, 'currently FuncDesigner cross(x,y) is implemented for arrays of length 3 only'
        return np.array([[0, -y[2], y[1]], [y[2], 0, -y[0]], [-y[1], y[0], 0]])
   
    r = oofun(lambda x, y: np.cross(x, y), [a, b], d=(lambda x, y: -aux_d(x, y), lambda x, y: aux_d(y, x)))
    r.getOrder = lambda *args, **kwargs: (inp1.getOrder(*args, **kwargs) if isinstance(inp1, oofun) else 0) + (inp2.getOrder(*args, **kwargs) if isinstance(inp2, oofun) else 0)
    return r

def ceil(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([ceil(elem) for elem in inp])        
    if not isinstance(inp, oofun): return np.ceil(inp)
    r = oofun(lambda x: np.ceil(x), inp, vectorized = True)
    r._D = lambda *args, **kwargs: raise_except('derivative for FD ceil is unimplemented yet')
    r.criticalPoints = False#lambda arg_infinum, arg_supremum: [np.ceil(arg_supremum)]
    return r

def floor(inp):
    if isinstance(inp, ooarray) and inp.dtype == object:
        return ooarray([floor(elem) for elem in inp])        
    if not isinstance(inp, oofun): return np.floor(inp)
    r = oofun(lambda x: np.floor(x), inp, vectorized = True)
    r._D = lambda *args, **kwargs: raise_except('derivative for FD floor is unimplemented yet')
    r.criticalPoints = False#lambda arg_infinum, arg_supremum: [np.floor(arg_infinum)]
    return r

def sum(inp, *args, **kwargs):
    if isinstance(inp, ooarray) and inp.dtype != object:
        inp = inp.view(np.ndarray)
        
    cond_ooarray = isinstance(inp, ooarray) and inp.dtype == object
    if cond_ooarray and inp.size == 1: 
        return np.asscalar(inp).sum()
    condIterableOfOOFuns = type(inp) in (list, tuple) or cond_ooarray
    
    if not isinstance(inp, oofun) and not condIterableOfOOFuns: 
        return np.sum(inp, *args, **kwargs)

    if isinstance(inp, ooarray) and inp.dtype == object: inp = inp.tolist()

    if condIterableOfOOFuns:
        d, INP, r0 = [], [], 0.0
        for elem in inp: # TODO: mb use reduce() or something like that
            if not isinstance(elem, oofun): 
                # not '+=' because size can be changed from 1 to another value
                r0 = r0 + np.asfarray(elem) # so it doesn't work for different sizes
                continue
            INP.append(elem)

        # TODO:  check for fixed inputs
        #f = lambda *args: r0 + np.sum(args)
        def f(*args):
            tmp = np.asarray(args)
            return r0 + (tmp.sum(0) if tmp.ndim > 1 else tmp.sum())
#            if np.asarray(args).size>12: raise 0
#            return r0 + np.sum(args)
#            

        
        #!!!!!!!!!!!!!!!!!! TODO: check INP for complex cases (not list of oovars)
        
#        if len(INP) == 0: 
#            INP = None
        r = oofun(f, INP) 
        
        def getOrder(*args, **kwargs):
            orders = [0]+[inp.getOrder(*args, **kwargs) for inp in INP]
            return np.max(orders)
        r.getOrder = getOrder
        
        def interval(domain, dtype):
            Arg_infinums, Arg_supremums = [], []
            for inp in INP:
                arg_inf, arg_sup = inp._interval(domain, dtype)
                Arg_infinums.append(arg_inf)
                Arg_supremums.append(arg_sup)
            #raise 0
            return r0+np.sum(np.vstack(Arg_infinums), 0), r0+np.sum(np.vstack(Arg_supremums), 0)
        r._interval = interval
        r.vectorized = True
        
        def _D(point, fixedVarsScheduleID, Vars=None, fixedVars = None, useSparse = 'auto'):
            # TODO: handle involvePrevData
            # TODO: handle fixed vars
            r, keys = {}, set()
            
            for elem in INP:
                if not elem.is_oovar and (elem.input is None or len(elem.input)==0 or elem.input[0] is None): 
                    continue # TODO: get rid if None, use [] instead
                if elem.discrete: continue
                if elem.is_oovar:
                    if (fixedVars is not None and elem in fixedVars) or (Vars is not None and elem not in Vars): continue
                    sz = np.asarray(point[elem]).size
                    tmpres = Eye(sz) 
                    if elem in keys:
                        if isinstance(r[elem], np.ndarray) and not isinstance(tmpres, np.ndarray): # i.e. tmpres is sparse matrix
                            tmpres = tmpres.toarray()
                        elif not isinstance(r[elem], np.ndarray) and isinstance(tmpres, np.ndarray):
                            r[elem] = r[elem].toarray()
                        r[elem] += tmpres
                    else:
                        # TODO: check it for oovars with size > 1
                        r[elem] = tmpres
                        keys.add(elem)
                else:
                    tmp = elem._D(point, fixedVarsScheduleID, Vars, fixedVars, useSparse = useSparse)
                    for key, val in tmp.items():
                        if key in keys:
                            if isinstance(r[key], np.ndarray) and not isinstance(val, np.ndarray): # i.e. tmpres is sparse matrix
                                val = val.toarray()
                            elif not isinstance(r[key], np.ndarray) and isinstance(val, np.ndarray):
                                r[key] = r[key].toarray()
                            try:
                                r[key] += val
                            except:
                                r[key] = r[key] + val
                        else:
                            r[key] = val
                            keys.add(key)
            if useSparse is False:
                for key, val in r.items():
                    if np.isscalar(val): val = np.asfarray(val)
                    if not isinstance(val, np.ndarray): # i.e. sparse matrix
                        r[key] = val.toarray()
            return r
        r._D = _D
        return r
    else: 
        return inp.sum(*args, **kwargs)#np.sum(inp, *args, **kwargs)
    
    if len(args) != 0 or len(kwargs) != 0:
        raise FuncDesignerException('oofun for sum(x, *args,**kwargs) is not implemented yet')
    return inp.sum()
    
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
    return sqrt(sum(args[0]**2),  attachConstraints=False)
    

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

def decision(*args, **kwargs):
    pass
        
def max(inp,  *args,  **kwargs): 
    if type(inp) in (list, tuple, np.ndarray) and (len(args) == 0 or len(args) == 1 and not isinstance(args[0], oofun)) and np.asarray(inp).dtype != object:
        return np.max(inp, *args, **kwargs)
    assert len(args) == len(kwargs) == 0, 'incorrect data type in FuncDesigner max or not implemented yet'
    
    if isinstance(inp, oofun):
        f = lambda x: np.max(x)
        def d(x):
            df = inp.d(x)
            ind = np.argmax(x)
            return df[ind, :]
        r = oofun(f, inp, d = d, size = 1)
    elif type(inp) in (list, tuple):
        f = lambda *args: np.max([arg for arg in args])
        r = oofun(f, inp, size = 1)
        def _D(point, *args, **kwargs):
            ind = np.argmax([(s(point) if isinstance(s, oofun) else s) for s in r.input])
            return r.input[ind]._D(point, *args, **kwargs) if isinstance(r.input[ind], oofun) else {}
        r._D = _D
    else:
        return np.max(inp, *args, **kwargs)
    return r        
    
def min(inp,  *args,  **kwargs): 
    if type(inp) in (list, tuple, np.ndarray) and (len(args) == 0 or len(args) == 1 and not isinstance(args[0], oofun)) and np.asarray(inp).dtype != object:
        return np.min(inp, *args, **kwargs)
    
    assert len(args) == len(kwargs) == 0, 'incorrect data type in FuncDesigner min or not implemented yet'
    if isinstance(inp, oofun):
        f = lambda x: np.min(x)
        def d(x):
            df = inp.d(x)
            #df = inp.d(x) if type(inp.d) not in (list, tuple) else np.hstack([item(x) for item in inp.d])
            ind = np.argmin(x)
            return df[ind, :]
        r = oofun(f, inp, d = d, size = 1)
    elif type(inp) in (list, tuple):
        f = lambda *args: np.min([arg for arg in args])
        r = oofun(f, inp, size = 1)
        def _D(point, *args, **kwargs):
            ind = np.argmin([(s(point) if isinstance(s, oofun) else s) for s in r.input])
            return r.input[ind]._D(point, *args, **kwargs) if isinstance(r.input[ind], oofun) else {}
        r._D = _D
    else:
        return np.min(inp, *args, **kwargs)
    return r        
    
#def fixed_oofun(Val):
#    val = np.asfarray(Val)
#    f = lambda: Val
#    r = oofun(f, input=[])
#    r._D = lambda *args,  **kwargs: {}
#    r.D = lambda *args,  **kwargs: {}
#    r.discrete = True
#    return r

det3 = lambda a, b, c: a[0] * (b[1]*c[2] - b[2]*c[1]) - a[1] * (b[0]*c[2] - b[2]*c[0]) + a[2] * (b[0]*c[1] - b[1]*c[0]) 

# TODO: move the func into fdmisc.py
def errFunc(*args,  **kwargs): 
    # this function shouldn't be ever called, an FD kernel hack has been involved
    raise FuncDesignerException('error in FuncDesigner kernel, inform developers')





#class oolist(list):
#    def __call__(self, *args, **kwargs):
#        #print 'ooarray call start'
#        tmp = [item(*args, **kwargs) if isinstance(item, oofun) else item for item in self]
#        r = oolist([np.asscalar(item) if type(item) in (np.ndarray, np.matrix) else item for item in tmp])
#        #print 'ooarray call end'
#        return r

#    def __getattr__(self, attr):
#        if attr == 'size':
