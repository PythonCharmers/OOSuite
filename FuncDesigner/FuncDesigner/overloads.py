from ooFun import oofun
import numpy as np
from misc import FuncDesignerException
import integrate 

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
#    return oofun(lambda x: np.sin(x), input = inp, d = d)

def sin(inp):
    if not isinstance(inp, oofun): return np.cos(inp)
    return oofun(np.sin, input = inp, d = lambda x: np.cos(x) if x.size == 1 else np.diag(np.cos(x)))

def cos(inp):
    if not isinstance(inp, oofun): return np.cos(inp)
    return oofun(np.cos, input = inp, d = lambda x: -np.sin(x) if x.size == 1 else np.diag(-np.sin(x)))

def tan(inp):
    if not isinstance(inp, oofun): return np.tan(inp)
    return oofun(np.tan, input = inp, d = lambda x: 1.0 / np.cos(x) ** 2 if x.size == 1 else np.diag(1.0 / np.cos(x) ** 2))

# TODO: cotan?

def arcsin(inp):
    if not isinstance(inp, oofun): return np.arcsin(inp)
    return oofun(np.arcsin, input = inp, d = lambda x: 1.0 / np.sqrt(1.0 - x**2) if x.size == 1 else np.diag(1.0 / np.sqrt(1.0 - x**2)))

def arccos(inp):
    if not isinstance(inp, oofun): return np.arccos(inp)
    return oofun(np.arccos, input = inp, d = lambda x: -1.0 / np.sqrt(1.0 - x**2) if x.size == 1 else np.diag(-1.0 / np.sqrt(1.0 - x**2)))

def arctan(inp):
    if not isinstance(inp, oofun): return np.arctan(inp)
    return oofun(np.arctan, input = inp, d = lambda x: 1.0 / (1.0 + x**2) if x.size == 1 else np.diag(1.0 / (1.0 + x**2)))

def sinh(inp):
    if not isinstance(inp, oofun): return np.sinh(inp)
    return oofun(np.sinh, input = inp, d = lambda x: np.cosh(x) if x.size == 1 else np.diag(np.cosh(x)))

def cosh(inp):
    if not isinstance(inp, oofun): return np.cosh(inp)
    return oofun(np.cosh, input = inp, d = lambda x: np.sinh(x) if x.size == 1 else np.diag(np.sinh(x)))

def exp(inp):
    if not isinstance(inp, oofun): return np.exp(inp)
    return oofun(np.exp, input = inp, d = lambda x: np.exp(x) if x.size == 1 else np.diag(np.exp(x)))

def sqrt(inp):
    if not isinstance(inp, oofun): return np.sqrt(inp)
    return oofun(np.sqrt, input = inp, d = lambda x: 0.5 / np.sqrt(x) if x.size == 1 else np.diag(0.5 / np.sqrt(x)))

def abs(inp):
    if not isinstance(inp, oofun): return np.abs(inp)
    return oofun(np.abs, input = inp, d = lambda x: np.sign(x)  if x.size == 1 else np.diag(np.sign(x)))    

def log(inp):
    if not isinstance(inp, oofun): return np.log(inp)
    return oofun(np.log, input = inp, d = lambda x: 1.0/x if x.size == 1 else np.diag(1.0/x))
    
def log10(inp):
    if not isinstance(inp, oofun): return np.log10(inp)
    return oofun(np.log10, input = inp, d = lambda x: 0.43429448190325176/x  if x.size == 1 else np.diag(0.43429448190325176/x))# 1 / (x * log_e(10))

def log2(inp):
    if not isinstance(inp, oofun): return np.log2(inp)
    return oofun(np.log2, input = inp, d = lambda x: 1.4426950408889634/x if x.size == 1 else np.diag(1.4426950408889634/x))# 1 / (x * log_e(2))

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
            if x.size != 1: r = np.diag(r)
        else:
            r = np.copy(y)
        return r
        
    r = oofun(lambda x, y: x * y if x.size == 1 or y.size == 1 else np.dot(x, y), input = [inp1, inp2], d=(lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x)))
    r.is_linear = is_linear
    return r

def sum(inp, *args, **kwargs):
    #return np.sum(inp, *args, **kwargs)
#    from time import time
#    t = time()
    
    # TODO: check for numpy.array of oofuns
    condIterableOfOOFuns = type(inp) in (list, tuple) and any([isinstance(elem, oofun) for elem in inp])
    
    if not isinstance(inp, oofun) and not condIterableOfOOFuns: 
        return np.sum(inp, *args, **kwargs)
        
    if condIterableOfOOFuns:
        d, INP, r0 = [], [], 0
        j = -1
        for elem in inp: # TODO: mb use reduce() or something like that
            if not isinstance(elem, oofun): 
                #r0 = r0 + elem # += is inappropriate because sizes may differ
                r0 += elem # so it doesn't work for different sizes
                continue
            j += 1
            if elem.is_oovar:
                # TODO: check it
                d.append(lambda *args: np.ones(len(args[j])))
                INP.append(elem)
            else:
                d.append(elem.D)
                INP.append(elem.input)
        
        # TODO:  check for fixed inputs
        f = lambda *args: r0 + np.sum(args)
        
        #!!!!!!!!!!!!!!!!!! TODO: check INP for complex cases (not list of oovars)
        r = oofun(f, d=tuple(d), input=INP) 
        return r
    else:
        return np.sum(inp, *args, **kwargs)
    
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
    return oofun(np.linalg.norm, input = inp, d = lambda x: d(x) if x.size == 1 else np.diag(d(x)))

def size(inp, *args, **kwargs):
    if not isinstance(inp, oofun): return np.size(inp, *args, **kwargs)
    return inp.size()
    
