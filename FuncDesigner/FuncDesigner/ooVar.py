# created by Dmitrey

from numpy import nan, asarray, isfinite, empty, zeros, inf, any, array, prod, atleast_1d, \
asfarray, isscalar, ndarray, int16, int32, int64, float64, tile, vstack, searchsorted, logical_or, where
from FDmisc import FuncDesignerException, checkSizes
from ooFun import oofun, Len, ooarray, BooleanOOFun, AND, OR, NOT
#from FuncDesigner.Interval import adjust_lx_WithDiscreteDomain, adjust_ux_WithDiscreteDomain

f_none = lambda *args, **kw: None
class oovar(oofun):
    is_oovar = True
    domain = None
    #shape = nan
    #fixed = False
    #initialized = False
    _unnamedVarNumber = 1#static variable for oovar class
    __hash__ = oofun.__hash__

    def __init__(self, name=None, *args, **kwargs):
        if len(args) > 0: raise FuncDesignerException('incorrect args number for oovar constructor')
        if name is None:
            self.name = 'unnamed_' + str(oovar._unnamedVarNumber)
            oovar._unnamedVarNumber += 1
        else:
            kwargs['name'] = name
        oofun.__init__(self, f_none, *args, **kwargs)
    
    def _interval_(self, domain, dtype = float64):
        tmp = domain.get(self, None)
        if tmp is None: return None
        if isinstance(tmp, ndarray) or isscalar(tmp): # thus variable value is fixed for this calculation
            tmp = asarray(tmp, dtype)
            return tile(tmp, (2, 1)), True
        infinum, supremum = tmp
        if type(infinum) in (list, tuple): 
            infinum = array(infinum, dtype)
        elif isscalar(infinum):
            infinum = dtype(infinum)
        if type(supremum) in (list, tuple): 
            supremum = array(supremum, dtype)
        elif isscalar(supremum):
            supremum = dtype(supremum)
            
#        if modificationVar is self:
#            assert dtype in (float, float64),  'other types unimplemented yet'
#            middle = 0.5 * (supremum+infinum)
#            return (vstack((infinum, middle)), True), (vstack((middle, supremum)), True)

        return vstack((infinum, supremum)), True
    
    #def _interval_(self, domain, dtype):
        #return self._interval()
    #_interval_ = _interval
    
    def _getFuncCalcEngine(self, x, **kwargs):
        if hasattr(x, 'xf'):return x.xf[self]
        r = x.get(self, None)
        if r is not None: 
            return r
        else:                             
            s = '''for oovar %s the point involved doesn't contain 
            neither name nor the oovar instance. 
            Maybe you try to get function value or derivative 
            in a point where value for an oovar is missing
            or run optimization problem 
            without setting initial value for this variable in start point
            ''' % self.name
            raise FuncDesignerException(s)
        
        
    def nlh(self, Lx, Ux, p, dataType, other=None):
        d = self.domain
        if d is None:
            raise FuncDesignerException('probably you are invoking boolean operation on continuous oovar')
        if d is int or d is 'int':
            raise FuncDesignerException('probably you are invoking boolean operation on non-boolean oovar')
        inds = p._oovarsIndDict.get(self, None)
        if inds is None:
            raise FuncDesignerException('probably you are trying to get nlh of fixed oovar, this is unimplemented in FD yet')
        ind1, ind2 = inds
        assert ind2-ind1 == 1, 'unimplemented for oovars of size > 1 yet'
        lx, ux = Lx[:, ind1], Ux[:, ind1]
        
#        from numpy import logical_or
#        assert all(logical_or(lx==1, lx==0))
#        assert all(logical_or(ux==1, ux==0))
        
        m = lx.size
        
        T0 = zeros(m)
        T2 = zeros((m, 2)) 
        
        if d is bool or d is 'bool':
            T0[ux != lx] = 0.5 # lx = 0, ux = 1
            T0[lx == 1.0] = 1.0 # lx = 1 => ux = 1
            T2[:, 0] = lx == 1
            T2[:, 1] = ux == 1
        else:
            assert other is not None, 'bug in FD kernel: called nlh with incorrect domain type'
            sd = d.size
            mx = 0.5 * (lx + ux)
            I = searchsorted(d, lx, 'left')
            I1 = searchsorted(d, mx, 'left')
            I2 = I1#searchsorted(d, mx, 'left')
            I3 = searchsorted(d, ux, 'left')
#            I1[I1==d.size] -= 1
#            I3[I3==d.size] -= 1
#            I = searchsorted(d, lx, 'left')
#            I1 = searchsorted(d, mx, 'left')
#            I2 = searchsorted(d, ux, 'left')
            
            d1, d2 = d[I], d[where(I1==sd, sd-1, I)]
            tmp = 1.0 / (I1-I+where(d2==other, 1, 0))
            tmp[logical_or(other<d1, other>d2)] = 0
            T2[:, 0] = tmp
            
            d1, d2 = d[I1], d[where(I3==sd, sd-1, I3)]
            tmp = 1.0 / (I3-I1+where(d2==other, 1, 0))
            tmp[logical_or(other<d1, other>d2)] = 0
            T2[:, 1] = tmp
            
            d1, d2 = d[I], d[where(I3==sd, sd-1, I3)]
            tmp = 1.0 / (I3-I+where(d2==other, 1, 0))
            tmp[logical_or(other<d1, other>d2)] = 0
            T0 = tmp

        res = {self:T2}
        DefiniteRange = True
        return T0, res, DefiniteRange
    
    __and__ = lambda self, other: AND(self, other)
    __or__ = lambda self, other: OR(self, other)
    __invert__ = NOT
    
#        if isinstance(x, dict):
#            tmp = x.get(self, None)
#            if tmp is not None:
#                return tmp #if type(tmp)==ndarray else asfarray(tmp)
#            elif self.name in x:
#                return asfarray(x[self.name])
#            else:
#                s = 'for oovar ' + self.name + \
#                " the point involved doesn't contain niether name nor the oovar instance. Maybe you try to get function value or derivative in a point where value for an oovar is missing"
#                raise FuncDesignerException(s)
#        elif hasattr(x, 'xf'):
#            # TODO: possibility of squeezing
#            return x.xf[self]
#        else:
#            raise FuncDesignerException('Incorrect data type (%s) while obtaining oovar %s value' %(type(x), self.name))
        
        
#    def _initialize(self, p):
#
#        """                                               Handling size and shape                                               """
#        sizes = set()
#        shapes = set()
#        for fn in ['v0', 'lb', 'ub']:
#            if hasattr(self, fn):
#                setattr(self, fn, asarray(getattr(self, fn)))
#                shapes.add(getattr(self, fn).shape)
#                sizes.add(getattr(self, fn).size)
#        if self.shape is not nan: 
#            shapes.add(self.shape)
#            sizes.add(prod(self.shape))
#        if self.size is not nan: sizes.add(self.size)
#        #if len(shapes) > 1: p.err('for oovar fields (if present) lb, ub, v0 should have same shape')
#        #elif len(shapes) == 1: self.shape = shapes.pop()
#        if len(shapes) >= 1: self.shape = prod(shapes.pop())
#        
#        if len(sizes) > 1: p.err('for oovar fields (if present) lb, ub, v0 should have same size')
#        elif len(sizes)==1 : self.size = sizes.pop()
#
#        if self.size is nan: self.size = asarray(self.shape).prod()
#        if self.shape is nan:
#            assert isfinite(self.size)
#            self.shape = (self.size, )
#        
#
#        """                                                     Handling init value                                                   """
##        if not hasattr(self, 'lb'):
##            self.lb = empty(self.shape)
##            self.lb.fill(-inf)
##        if not hasattr(self, 'ub'):
##            self.ub = empty(self.shape)
##            self.ub.fill(inf)
##        if any(self.lb > self.ub):
##            p.err('lower bound exceeds upper bound, solving impossible')
#        if not hasattr(self, 'v0'):
#            #p.warn('got oovar w/o init value')
#            v0 = zeros(self.shape)
#
#            ind = isfinite(self.lb) & isfinite(self.ub)
#            v0[ind] = 0.5*(self.lb[ind] + self.ub[ind])
#
#            ind = isfinite(self.lb) & ~isfinite(self.ub)
#            v0[ind] = self.lb[ind]
#
#            ind = ~isfinite(self.lb) & isfinite(self.ub)
#            v0[ind] = self.ub[ind]
#
#            self.v0 = v0
#            
#        self.initialized = True
        
        
def oovars(*args, **kw):
    
    if len(args) == 1:
        if type(args[0]) in (int, int16, int32, int64):
            return ooarray([oovar(**kw) for i in range(args[0])])
        elif type(args[0]) in [list, tuple]:
            return ooarray([oovar(name=args[0][i], **kw) for i in range(len(args[0]))])
        elif type(args[0]) == str:
            return ooarray([oovar(name=s, **kw) for s in args[0].split()])
        else:
            raise FuncDesignerException('incorrect args number for oovars constructor')
    else:
        return ooarray([oovar(name=args[i], **kw) for i in range(len(args))])



