# created by Dmitrey

#from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, \
#where, array, nan, ix_, vstack, eye, array_equal, isscalar, diag, log, hstack, sum, prod, nonzero, isnan
#from numpy.linalg import norm
#from misc import FuncDesignerException, Diag, Eye, pWarn, scipyAbsentMsg
#from copy import deepcopy

from misc import FuncDesignerException
from numpy import asfarray, ndarray, isscalar

Len = lambda x: 1 if isscalar(x) else x.size if type(x)==ndarray else len(x)

class ooPoint(dict):
    _id = 0
    def __init__(self, *args, **kwargs):
        if args:
            if not isinstance(args[0], dict):
                items = [(key, asfarray(val) if not isscalar(val) else float(val)) for key, val in args[0]] 
            else:
                items = [(key, asfarray(val) if not isscalar(val) else float(val)) for key, val in args[0].items()] 
        elif kwargs:
            items = [(key, asfarray(val) if not isscalar(val) else float(val)) for key, val in kwargs.items()]
        else:
            raise FuncDesignerException('incorrect oopoint constructor arguments')
            
        dict.__init__(self, items)
        
        for key, val in items:
            #assert type(val) not in [list, ndarray] or type(val[0]) != int
            if 'size' in key.__dict__ and type(key.size) == int and Len(val)  != key.size: 
                s = 'incorrect size for oovar %s: %d is required, %d is obtained' % (self.name, self.size, Size)
                raise FuncDesignerException(s)
        
        ooPoint._id += 1
        self._id = ooPoint._id
    
    def __setitem__(self, *args, **kwargs):
        raise FuncDesignerException('ooPoint must be immutable')
        
