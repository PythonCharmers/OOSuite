import numpy as np, operator
from numpy import ndarray
from FuncDesigner.FDmisc import FuncDesignerException
from baseClasses import MultiArray

if 'div' in operator.__dict__:
    div = operator.div
else:
    div = operator.truediv

# TODO: rework buggy multiarray.size
#delattr(multiarray, 'size')

class multiarray(MultiArray):
    __array_priority__ = 5
    __add__ = lambda self, other: multiarray_op(self, other, operator.add)
    __radd__ = lambda self, other: self.__add__(other)
    
    __sub__ = lambda self, other: multiarray_op(self, other, operator.sub)
    __rsub__ = lambda self, other: multiarray_op(-self, other, operator.add)
    
    
    __mul__ = lambda self, other: multiarray_op(self, other, operator.mul)
    __rmul__ = lambda self, other: self.__mul__(other)
    __div__ = lambda self, other: multiarray_op(self, other, div)
    __rdiv__ = lambda self, other: multiarray_op(other, self, div)
    
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    # TODO: rdiv, rpow
    __pow__ = lambda self, other: multiarray_op(self, other, operator.pow)
    __rpow__ = lambda self, other: multiarray_op(other, self, operator.pow)
    
    def __getitem__(self, ind): 
        return self.view(np.ndarray)[:, ind].view(multiarray)  if type(ind) in (int, np.int32, np.int64, np.int16, np.int8) \
        else self.__getslice__(ind.start, ind.stop) if type(ind) != tuple \
        else self.__getslice__(ind[0], ind[1])
        
        
    def __getslice__(self, ind1, ind2):
        #TODO: mb check if size is known then use it instead of None?
        cond_1 = ind1 is None
        cond_2 = ind2 is None or (type(ind2) == slice and ind2.start is None and ind2.stop is None and ind2.step is None)
        if cond_1 and cond_2:
            return self
            
        if cond_1: 
            ind1 = 0
        if cond_2: 
            ind2 = self.shape[1]
            
        return self.view(np.ndarray)[:, ind1:ind2].view(multiarray)
        
        
    # TODO: check it!
    #toarray = lambda self: self.view(ndarray)

    def sum(self, *args, **kw):
        if any([v is not None for v in args]): # somehow triggered from pswarm
            raise FuncDesignerException('arguments for FD multiarray sum are not implemened yet')
        if any([v is not None for v in kw.values()]):
            raise FuncDesignerException('keyword arguments for FD multiarray sum are not implemened yet')
        return np.sum(np.atleast_2d(self.view(ndarray)), 0).view(multiarray)

def multiarray_op(x, y, op):
    if isinstance(y, multiarray):
        if isinstance(x, multiarray):
            if x.size == y.size:
                r = op(x.view(ndarray), y.view(ndarray))
            else:
                assert x.ndim <= 1 or y.ndim <= 1, 'unimplemented yet'
                assert x.ndim < 3 and y.ndim < 3, 'unimplemented yet'
                X, Y = np.atleast_2d(x).view(ndarray), np.atleast_2d(y).view(ndarray)
                r = op(X, Y)
                #r = multiarray([op(x[i], y[i]) for i, X in enumerate(x)])
        else:
            r = op(x.reshape(-1, 1) if isinstance(x, ndarray) and x.size != 1 else x, y.view(ndarray))
    elif isinstance(x, multiarray): # and y is not multiarray here
        r = op(x.view(ndarray), y.reshape(-1, 1) if isinstance(y, ndarray) and y.size != 1 else y)
    else: # neither x nor y are multiarrays
        raise FuncDesignerException('bug in FuncDesigner kernel')
    return r.view(multiarray)#.flatten()


