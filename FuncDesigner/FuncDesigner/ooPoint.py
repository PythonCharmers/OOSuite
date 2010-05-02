# created by Dmitrey

#from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, \
#where, array, nan, ix_, vstack, eye, array_equal, isscalar, diag, log, hstack, sum, prod, nonzero, isnan
#from numpy.linalg import norm
#from misc import FuncDesignerException, Diag, Eye, pWarn, scipyAbsentMsg
#from copy import deepcopy

class ooPoint(dict):
    _id = 0
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        ooPoint._id += 1
        self._id = ooPoint._id
        
