# created by Dmitrey

#from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, \
#where, array, nan, ix_, vstack, eye, array_equal, isscalar, diag, log, hstack, sum, prod, nonzero, isnan
#from numpy.linalg import norm
#from misc import FuncDesignerException, Diag, Eye, pWarn, scipyAbsentMsg
#from copy import deepcopy

from FDmisc import FuncDesignerException
from baseClasses import Stochastic
from numpy import asanyarray, ndarray, isscalar
try:
    from scipy.sparse import isspmatrix
except ImportError:
    isspmatrix = lambda *args, **kw: False

Len = lambda x: 1 if isscalar(x) else x.size if type(x)==ndarray else len(x)

   

def ooMultiPoint(*args, **kw):
    kw['skipArrayCast'] = True
    r = ooPoint(*args, **kw)
    r.isMultiPoint = True
    return r

class ooPoint(dict):
    _id = 0
    isMultiPoint = False
    modificationVar = None # default: no modification variable
    useSave = True
    useAsMutable = False
    
    def __init__(self, *args, **kwargs):
        self.storedIntervals = {}
        self.storedSums = {}
        self.dictOfFixedFuncs = {}
        
        for fn in ('isMultiPoint', 'modificationVar', 'useSave', 'useAsMutable', 'maxDistributionSize'):
            tmp = kwargs.get(fn, None)
            if tmp is not None:
                setattr(self, fn, tmp)
        
        if kwargs.get('skipArrayCast', False): 
            Asanyarray = lambda arg: arg
        else: 
            Asanyarray = lambda arg: asanyarray(arg)  if not isinstance(arg, Stochastic) else arg#if not isspmatrix(arg) else arg
            
        # TODO: remove float() after Python 3 migraion
        if args:
            if not isinstance(args[0], dict):
                items = [(key, Asanyarray(val) if not isscalar(val) else float(val) if type(val) == int else val) for key, val in args[0]] 
            else:
                items = [(key, Asanyarray(val) if not isscalar(val) else float(val) if type(val) == int else val) for key, val in args[0].items()] 
        elif kwargs:
            items = [(key, Asanyarray(val) if not isscalar(val) else float(val) if type(val) == int else val) for key, val in kwargs.items()]
        else:
            raise FuncDesignerException('incorrect oopoint constructor arguments')
            
        dict.__init__(self, items)

# TODO: fix it wrt ode2.py

#        for key, val in items:
#            #assert type(val) not in [list, ndarray] or type(val[0]) != int
#            if 'size' in key.__dict__ and type(key.size) == int and Len(val)  != key.size: 
#                s = 'incorrect size for oovar %s: %d is required, %d is obtained' % (key, self.size, Size)
#                raise FuncDesignerException(s)
        
        ooPoint._id += 1
        self._id = ooPoint._id
    
    def __setitem__(self, *args, **kwargs):
        if not self.useAsMutable:
            raise FuncDesignerException('ooPoint must be immutable')
        dict.__setitem__(self, *args, **kwargs)
        
