# created by Dmitrey

from numpy import nan, asarray, isfinite, empty, zeros, inf, any, array, prod, atleast_1d, \
asfarray, isscalar, ndarray, int16, int32, int64, float64, tile, vstack, searchsorted, logical_or, where, \
asanyarray, string_, arange, log2, logical_and
from FDmisc import FuncDesignerException, checkSizes
from ooFun import oofun, Len, ooarray, BooleanOOFun, AND, OR, NOT, IMPLICATION, EQUIVALENT
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
        r = x.get(self.name, None)
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
        DefiniteRange = True
        d = self.domain
        if d is None:
            raise FuncDesignerException('probably you are invoking boolean operation on continuous oovar')
        if d is int or d is 'int':
            raise FuncDesignerException('probably you are invoking boolean operation on non-boolean oovar')
        inds = p._oovarsIndDict.get(self, None)
        m = Lx.shape[0]
        if inds is None:
            # this oovar is fixed
            res = {}
            if self.domain is bool or self.domain is 'bool':
                T0 = True if p._x0[self] == 1 else False # 0 or 1
            else:
                assert other is not None, 'bug in FD kernel: called nlh with incorrect domain type'
                T0 = False if p._x0[self] != other else True
            return T0, res, DefiniteRange
            #raise FuncDesignerException('probably you are trying to get nlh of fixed oovar, this is unimplemented in FD yet')
        ind1, ind2 = inds
        assert ind2-ind1 == 1, 'unimplemented for oovars of size > 1 yet'
        lx, ux = Lx[:, ind1], Ux[:, ind1]
        
#        from numpy import logical_or
#        assert all(logical_or(lx==1, lx==0))
#        assert all(logical_or(ux==1, ux==0))
        
        #m = lx.size
        
        if d is bool or d is 'bool':
            T0 = empty(m)
            T0.fill(inf)
            T2 = empty((m, 2)) 
            T2.fill(inf)
        
            T0[ux != lx] = 1.0 # lx = 0, ux = 1 => -log2(0.5) = 1
            T0[lx == 1.0] = 0.0 # lx = 1 => ux = 1 => -log2(1) = 0
            T2[:, 0] = where(lx == 1, 0, inf)
            T2[:, 1] = where(ux == 1, 0, inf)
        else:
            assert other is not None, 'bug in FD kernel: called nlh with incorrect domain type'
            T2 = empty((m, 2)) 
            sd = d.size
            mx = 0.5 * (lx + ux) 

            ind = logical_and(mx==other, lx != ux)
            if any(ind):
                mx[ind] -= 1e-15 + 1e-15*abs(mx[ind])

            prev = 0
            if prev:
                I = searchsorted(d, lx, 'left')
                J = searchsorted(d, mx, 'left')
                K = searchsorted(d, ux, 'left')
            else:
                I = searchsorted(d, lx, 'right') -  1
                J = searchsorted(d, mx, 'right') - 1
                K = searchsorted(d, ux, 'right') - 1
            D0, D1, D2 = d[I], d[J], d[K]
            
            d1, d2 = D0, D1
            tmp = asfarray(J-I+where(d2==other, 1, 0))
            tmp[logical_or(other<d1, other>d2)] = inf
            
#            ind = tmp == 0            
#            if any(logical_and(ind, logical_or(other>mx, other < lx))):
#                print '1!', other, tmp[ind]

#            if any(ind) and (any(tmp[ind]==0) or any(tmp[ind] == inf)):
#                print '1:', tmp[ind], lx[ind], ux[ind], other
            T2[:, 0] = tmp
            
            
            d1, d2 = D1, D2
            tmp =  asfarray(K-J+where(d2==other, 1, 0))
            tmp[logical_or(other<d1, other>d2)] = inf
#            ind = tmp == 0
#            if any(logical_and(ind, logical_or(other>ux, other < mx))):
#                print '2!', other, tmp[ind]
#            if any(ind) and (any(tmp[ind]==0) or any(tmp[ind] == inf)):
#                print '2:', tmp[ind], lx[ind], ux[ind], other
            T2[:, 1] = tmp
            
            T2 = log2(T2)
            
            d1, d2 = D0, D2
            tmp = asfarray(K-I+where(d2==other, 1, 0))
            tmp[logical_or(other<d1, other>d2)] = inf
#            ind = tmp == 0
#            if any(logical_and(ind, logical_or(other>ux, other < lx))):
#                print '3!', other, tmp[ind]
#            if any(ind) and (any(tmp[ind]==0) or any(tmp[ind] == inf)):
#                print '3:', tmp[ind], lx[ind], ux[ind], other
            T0 = log2(tmp)

        res = {self:T2}
#        if all(lx == ux): 
#            if other == lx[0]:
#                print '1!', T0, res
#            else:
#                print '2!', T0, res
        return T0, res, DefiniteRange
    
    __and__ = AND
    __or__ = OR
    implication = IMPLICATION
    __invert__ = NOT
    __ne__ = lambda self, arg: NOT(self==arg)
    def __eq__(self, other): 
        if (self.domain is bool or self.domain is 'bool') and isinstance(other, (oovar, BooleanOOFun)):
            return EQUIVALENT(self, other)
        else:
            return oofun.__eq__(self, other)
    
    def formAuxDomain(self):
        if 'aux_domain' in self.__dict__: return
        self.domain = asanyarray(self.domain)
        d = self.domain
        if d.dtype.type not in [string_, unicode, str]:
            raise FuncDesignerException('to compare string with oovar latter should have domain of string type')
        if any(d[1:] < d[:-1]):
            d.sort()
        self.domain, self.aux_domain = arange(d.size), d    
    
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



