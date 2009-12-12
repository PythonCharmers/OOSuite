# created by Dmitrey

from numpy import nan, asarray, isfinite, empty, zeros, inf, any, array, prod, atleast_1d, asfarray, isscalar
from misc import FuncDesignerException, checkSizes
from ooFun import oofun

class oovar(oofun):
    #size = nan # number of variables
    is_oovar = True
    shape = nan
    fixed = False
    initialized = False
    is_linear = True
    _unnamedVarNumber = 1#static variable for oovar class
    #size = None # no requirements for size

    def __init__(self, name=None, *args, **kwargs):
        if name is None:
            self.name = 'unnamed_' + str(oovar._unnamedVarNumber)
            oovar._unnamedVarNumber += 1
        else:
            self.name = name
        
        if len(args) > 0: raise FuncDesignerException('incorrect args number for oovar constructor')

        if 'size' in kwargs.keys():
            self.size = kwargs['size']
        elif not 'skipRecursive' in kwargs.keys() or kwargs['skipRecursive'] == False:
            self.size = oofun(lambda x: asarray(x).size, input = self, is_linear = True, discrete = True, skipRecursive = True)

        for key in kwargs.keys():
            if key != 'skipRecursive':
                setattr(self, key, kwargs[key])

    def __repr__(self):
        return self.name
    
    def requireSize(self, size):
        self.size = size

    def _getFunc(self, x):
        try:
            r = atleast_1d(asfarray(x[self])) if self in x else atleast_1d(asfarray(x[self.name]))
        except KeyError:
            s = 'for oovar ' + self.name + \
            " the point involved doesn't contain niether name nor the oovar instance. Maybe you try to get function value or derivative in a point where value for an oovar is missing"
            raise FuncDesignerException(s)
        Size = r.size
        if isscalar(self.size) and Size != self.size:
            s = 'incorrect size for oovar %s: %d is required, %d is obtained' % (self.name, self.size, Size)
            raise FuncDesignerException(s)
        return r
        
#        self_in_keys = self in x.keys()
#        self_name_in_keys = self.name in x.keys()
#        if self_in_keys and self_name_in_keys:
#            raise FuncDesignerException('for oovar ' + self.name + ' the point involved contains both name and the oovar instance, that is forbidden because of ambiguity')
#        elif not self_in_keys and not self_name_in_keys:
#            raise FuncDesignerException('for oovar ' + self.name + " the point involved doesn't contain niether name nor the oovar instance")
#        elif self_in_keys:
#            return atleast_1d(asfarray(x[self]))
#        else: # self_name_in_keys
#            return atleast_1d(asfarray(x[self.name]))
        
            
    def _initialize(self, p):

        """                                               Handling size and shape                                               """
        sizes = set([])
        shapes = set([])
        for fn in ['v0', 'lb', 'ub']:
            if hasattr(self, fn):
                setattr(self, fn, asarray(getattr(self, fn)))
                shapes.add(getattr(self, fn).shape)
                sizes.add(getattr(self, fn).size)
        if self.shape is not nan: 
            shapes.add(self.shape)
            sizes.add(prod(self.shape))
        if self.size is not nan: sizes.add(self.size)
        #if len(shapes) > 1: p.err('for oovar fields (if present) lb, ub, v0 should have same shape')
        #elif len(shapes) == 1: self.shape = shapes.pop()
        if len(shapes) >= 1: self.shape = prod(shapes.pop())
        
        if len(sizes) > 1: p.err('for oovar fields (if present) lb, ub, v0 should have same size')
        elif len(sizes)==1 : self.size = sizes.pop()

        if self.size is nan: self.size = asarray(self.shape).prod()
        if self.shape is nan:
            assert isfinite(self.size)
            self.shape = (self.size, )
        

        """                                                     Handling init value                                                   """
#        if not hasattr(self, 'lb'):
#            self.lb = empty(self.shape)
#            self.lb.fill(-inf)
#        if not hasattr(self, 'ub'):
#            self.ub = empty(self.shape)
#            self.ub.fill(inf)
#        if any(self.lb > self.ub):
#            p.err('lower bound exceeds upper bound, solving impossible')
        if not hasattr(self, 'v0'):
            #p.warn('got oovar w/o init value')
            v0 = zeros(self.shape)

            ind = isfinite(self.lb) & isfinite(self.ub)
            v0[ind] = 0.5*(self.lb[ind] + self.ub[ind])

            ind = isfinite(self.lb) & ~isfinite(self.ub)
            v0[ind] = self.lb[ind]

            ind = ~isfinite(self.lb) & isfinite(self.ub)
            v0[ind] = self.ub[ind]

            self.v0 = v0
            
        self.initialized = True
        
        
def oovars(*args, **kwargs):
    assert len(kwargs) ==0
    if len(args) == 1:
        if isinstance(args[0], int):
            return [oovar() for i in xrange(args[0])]
        elif type(args[0]) in [list, tuple]:
            return [oovar(name=args[0][i]) for i in xrange(len(args[0]))]
    else:
        return [oovar(name=args[i]) for i in xrange(len(args))]



