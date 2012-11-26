import numpy as np, operator
from numpy import ndarray
from FuncDesigner.FDmisc import FuncDesignerException
from baseClasses import MultiArray

if 'div' in operator.__dict__:
    div = operator.div
else:
    div = operator.truediv

class multiarray(MultiArray):
    __array_priority__ = 5
    __add__ = lambda self, other: multiarray_op(self, other, operator.add)
    __radd__ = lambda self, other: self.__add__(other)
    __mul__ = lambda self, other: multiarray_op(self, other, operator.mul)
    __rmul__ = lambda self, other: self.__mul__(other)
    __div__ = lambda self, other: multiarray_op(self, other, div)
    __truediv__ = __div__
    # TODO: rdiv, rpow
    __pow__ = lambda self, other: multiarray_op(self, other, operator.pow)
    __rpow__ = lambda self, other: multiarray_op(other, self, operator.pow)
    
    # TODO: check it!
    toarray = lambda self: self.view(ndarray)

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
    
