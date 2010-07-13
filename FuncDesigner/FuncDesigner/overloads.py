from ooFun import oofun
import numpy as np
from misc import FuncDesignerException, Diag, Eye

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
    
    if not isinstance(inp1, oofun): 
        #inp1, inp2 = inp2, np.asfarray(inp1)
        is_linear = inp2.is_linear
    else:
        is_linear = inp1.is_linear and not isinstance(inp2, oofun)
    
    def aux_d(x, y):
        #assert x.ndim <= 1 and y.ndim <= 1
        if y.size == 1: 
            r = np.empty(x.size)
            r.fill(y)
            r = Diag(r)
        else:
            r = np.copy(y)
        return r
        
    r = oofun(lambda x, y: x * y if x.size == 1 or y.size == 1 else np.dot(x, y), [inp1, inp2], d=(lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x)))
    r.is_linear = is_linear
    r.isCostly = True
    return r


def sum(inp, *args, **kwargs):
    #return np.sum(inp, *args, **kwargs)
#    from time import time
#    t = time()
    
    # TODO: check for numpy.array of oofuns
    #condIterableOfOOFuns = type(inp) in (list, tuple) and any([isinstance(elem, oofun) for elem in inp])
    condIterableOfOOFuns = type(inp) in (list, tuple) and any([isinstance(elem, oofun) for elem in inp])
    
    if not isinstance(inp, oofun) and not condIterableOfOOFuns: 
        return np.sum(inp, *args, **kwargs)

    if condIterableOfOOFuns:
        is_linear = True
        d, INP, r0 = [], [], 0
        j = -1
        for elem in inp: # TODO: mb use reduce() or something like that
            if not isinstance(elem, oofun): 
                #r0 = r0 + elem # += is inappropriate because sizes may differ
                r0 += np.asfarray(elem) # so it doesn't work for different sizes
                continue
            j += 1
            INP.append(elem)
            if not elem.is_linear: is_linear = False

        # TODO:  check for fixed inputs
        f = lambda *args: r0 + np.sum(args)
#        def f(*args):
#            print args
#            return np.sum(args)
        _inp = set(INP)
        #!!!!!!!!!!!!!!!!!! TODO: check INP for complex cases (not list of oovars)
        r = oofun(f, INP, is_linear=is_linear) 
        def _D(point, diffVarsID, Vars=None, fixedVars = None, asSparse = 'auto'):
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
                    tmp = elem._D(point, diffVarsID, Vars, fixedVars, *args, **kwargs)
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
            if asSparse is False:
                for key, val in r.iteritems():
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
    return inp.prod()

def norm(inp, *args, **kwargs):
    if len(args) != 0 or len(kwargs) != 0:
        raise FuncDesignerException('oofun for norm(x, *args,**kwargs) is not implemented yet')
    def d(x):
        s = np.sqrt(np.sum(x**2))
        return x /  s if s != 0 else np.zeros(x.size) # however, dirivative doesn't exist in (0,0,..., 0)
    return oofun(np.linalg.norm, inp, d = lambda x: Diag(d(x)), isCostly = True)

def size(inp, *args, **kwargs):
    if not isinstance(inp, oofun): return np.size(inp, *args, **kwargs)
    return inp.size()
    
def ifThenElse(condition, val1, val2, *args, **kwargs):
    
    # for future implementation
    assert len(args) == 0  
    assert len(kwargs) == 0 
    Val1 = fixed_oofun(val1) if not isinstance(val1, oofun) else val1
    Val2 = fixed_oofun(val2) if not isinstance(val2, oofun) else val2
    if isinstance(condition, bool): 
        return Val1 if condition else Val2
    elif isinstance(condition, oofun):
            
        #def f(conditionResult, value1Result, value2Result): 
            #return value1Result if conditionResult else value2Result
        f = lambda point: (Val1(point) if isinstance(Val1, oofun) else Val1) if condition(point) else (Val2(point) if isinstance(Val2, oofun) else Val2)
        
        
        # !!! Don't modify it elseware function will evaluate both expressions despite of condition value 
        r = oofun(errFunc, [condition, val1, val2])
        r._getFunc = f
        r.D = lambda point, *args, **kwargs: (Val1.D(point, *args, **kwargs) if isinstance(Val1, oofun) else {}) if condition(point) else \
        (Val2.D(point, *args, **kwargs) if isinstance(Val2, oofun) else {})
        r._D = lambda point, *args, **kwargs: (Val1._D(point, *args, **kwargs) if isinstance(Val1, oofun) else {}) if condition(point) else \
        (Val2._D(point, *args, **kwargs) if isinstance(Val2, oofun) else {})
        r.d = errFunc
        return r
    else:
        raise FuncDesignerException('ifThenElse requires 1st argument (condition) to be either boolean or oofun, got %s instead' % type(condition))
        
def fixed_oofun(Val):
    val = np.asfarray(Val)
    f = lambda: Val
    r = oofun(f, input=[])
    r._D = lambda *args,  **kwargs: {}
    r.D = lambda *args,  **kwargs: {}
    r.discrete = True
    return r

# TODO: move the func into fdmisc.py
def errFunc(*args,  **kwargs): 
    # this function shouldn't be ever called, an FD kernel hack has been involved
    raise FuncDesignerException('error in FuncDesigner kernel, inform developers')

