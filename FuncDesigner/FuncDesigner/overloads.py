from ooFun import oofun
import numpy as np
from misc import FuncDesignerException, Diag, Eye
from ooFun import atleast_oofun

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
    if not isinstance(inp, oofun): return np.sin(inp)
    return oofun(np.sin, inp, d = lambda x: Diag(np.cos(x)))

def cos(inp):
    if not isinstance(inp, oofun): return np.cos(inp)
    return oofun(np.cos, inp, d = lambda x: Diag(-np.sin(x)))

def tan(inp):
    if not isinstance(inp, oofun): return np.tan(inp)
    return oofun(np.tan, inp, d = lambda x: Diag(1.0 / np.cos(x) ** 2))

# TODO: cotan?

def arcsin(inp):
    if not isinstance(inp, oofun): return np.arcsin(inp)
    r = oofun(np.arcsin, inp, d = lambda x: Diag(1.0 / np.sqrt(1.0 - x**2)))
    r.attach((inp>-1)('arcsin_domain_lower_bound_%d' % r._id, tol=-1e-7), (inp<1)('arcsin_domain_upper_bound_%d' % r._id, tol=-1e-7))
    return r

def arccos(inp):
    if not isinstance(inp, oofun): return np.arccos(inp)
    r = oofun(np.arccos, inp, d = lambda x: Diag(-1.0 / np.sqrt(1.0 - x**2)))
    r.attach((inp>-1)('arccos_domain_lower_bound_%d' % r._id, tol=-1e-7), (inp<1)('arccos_domain_upper_bound_%d' % r._id, tol=-1e-7))
    return r

def arctan(inp):
    if not isinstance(inp, oofun): return np.arctan(inp)
    return oofun(np.arctan, inp, d = lambda x: Diag(1.0 / (1.0 + x**2)))

def sinh(inp):
    if not isinstance(inp, oofun): return np.sinh(inp)
    return oofun(np.sinh, inp, d = lambda x: Diag(np.cosh(x)))

def cosh(inp):
    if not isinstance(inp, oofun): return np.cosh(inp)
    return oofun(np.cosh, inp, d = lambda x: Diag(np.sinh(x)))

def exp(inp):
    if not isinstance(inp, oofun): return np.exp(inp)
    return oofun(np.exp, inp, d = lambda x: Diag(np.exp(x)))

def sqrt(inp):
    if not isinstance(inp, oofun): return np.sqrt(inp)
    r = oofun(np.sqrt, inp, d = lambda x: Diag(0.5 / np.sqrt(x)))
    r.attach((inp>0)('sqrt_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

def abs(inp):
    if not isinstance(inp, oofun): return np.abs(inp)
    return oofun(np.abs, inp, d = lambda x: Diag(np.sign(x)))    

def log(inp):
    if not isinstance(inp, oofun): return np.log(inp)
    r = oofun(np.log, inp, d = lambda x: Diag(1.0/x))
    r.attach((inp>1e-300)('log_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r
    
def log10(inp):
    if not isinstance(inp, oofun): return np.log10(inp)
    r = oofun(np.log10, inp, d = lambda x: Diag(0.43429448190325176/x))# 1 / (x * log_e(10))
    r.attach((inp>1e-300)('log10_domain_zero_bound_%d' % r._id, tol=-1e-7))
    return r

def log2(inp):
    if not isinstance(inp, oofun): return np.log2(inp)
    r = oofun(np.log2, inp, d = lambda x: Diag(1.4426950408889634/x))# 1 / (x * log_e(2))
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


def _sum(inp, *args, **kwargs):
    #return np.sum(inp, *args, **kwargs)
#    from time import time
#    t = time()
    
    # TODO: check for numpy.array of oofuns
    #condIterableOfOOFuns = type(inp) in (list, tuple) and any([isinstance(elem, oofun) for elem in inp])
    condIterableOfOOFuns = type(inp) in (list, tuple) and any([isinstance(elem, oofun) for elem in inp])
    
    if not isinstance(inp, oofun) and not condIterableOfOOFuns: 
        return np.sum(inp, *args, **kwargs)

    if condIterableOfOOFuns:
        d, INP, r0 = [], [], 0.0
        j = -1
        for elem in inp: # TODO: mb use reduce() or something like that
            if not isinstance(elem, oofun): 
                #r0 = r0 + elem # += is inappropriate because sizes may differ
                
                # not '+=' because size can be changed from 1 to another value
                r0 = r0 + np.asfarray(elem) # so it doesn't work for different sizes
                
                continue
                
            j += 1
            INP.append(elem)

        # TODO:  check for fixed inputs
        f = lambda *args: r0 + sum(args)
#        def f(*args):
#            print args
#            return np.sum(args)
        _inp = set(INP)
        #!!!!!!!!!!!!!!!!!! TODO: check INP for complex cases (not list of oovars)
        
        r = oofun(f, INP) 
        
        def getOrder(*args, **kwargs):
            orders = [0]+[inp.getOrder(*args, **kwargs) for inp in INP]
            return max(orders)
        r.getOrder = getOrder
        
        def _D(point, fixedVarsScheduleID, Vars=None, fixedVars = None, useSparse = 'auto'):
            # TODO: handle involvePrevData
            # TODO: handle fixed vars
            r, keys = {}, set()
            
            for elem in _inp:
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
                    tmp = elem._D(point, fixedVarsScheduleID, Vars, fixedVars, *args, **kwargs)
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
norm = lambda inp: sqrt(inp**2)

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
    assert len(args) == 0  
    assert len(kwargs) == 0 
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
        
        
def max(inp,  *args,  **kwargs): 
    assert len(args) == len(kwargs) == 0, 'incorrect data type in FuncDesigner max or not implemented yet'
    
    if isinstance(inp, oofun):
        f = lambda x: np.max(x)
        def d(x):
            df = inp.d(x)
            ind = np.argmax(x)
            return df[ind, :]
    elif type(inp) in (list, tuple):
        f = lambda *args: np.max([arg for arg in args])
        def d(*args):
            #df = asfarray([arg.d(x) for arg in inp]
            ind = np.argmax(args)
            raise 'not implemented yet'
            return inp[ind].d(args)
    
    else:
        raise FuncDesignerException('incorrect data type in FuncDesigner max')
            
    r = oofun(f, inp, d = d, size = 1)
    return r        
    
def min(inp,  *args,  **kwargs): 
    assert len(args) == len(kwargs) == 0, 'incorrect data type in FuncDesigner min or not implemented yet'
    
    if isinstance(inp, oofun):
        f = lambda x: np.min(x)
        def d(x):
            df = inp.d(x)
            #df = inp.d(x) if type(inp.d) not in (list, tuple) else np.hstack([item(x) for item in inp.d])
            ind = np.argmin(x)
            return df[ind, :]
    elif type(inp) in (list, tuple):
        f = lambda *args: np.min([arg for arg in args])
        def d(*args):
            ind = np.argmin(args)
            raise 'not implemented yet'
            return inp[ind].d(args)
    else:
        raise FuncDesignerException('incorrect data type in FuncDesigner min')
            
    r = oofun(f, inp, d = d, size = 1)
    return r        
    
#def fixed_oofun(Val):
#    val = np.asfarray(Val)
#    f = lambda: Val
#    r = oofun(f, input=[])
#    r._D = lambda *args,  **kwargs: {}
#    r.D = lambda *args,  **kwargs: {}
#    r.discrete = True
#    return r

# TODO: move the func into fdmisc.py
def errFunc(*args,  **kwargs): 
    # this function shouldn't be ever called, an FD kernel hack has been involved
    raise FuncDesignerException('error in FuncDesigner kernel, inform developers')

