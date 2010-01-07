# created by Dmitrey

from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, \
where, isfinite, array, nan, ix_, vstack, eye, array_equal, isscalar, diag, log, hstack, sum, prod, nonzero
from numpy.linalg import norm
from misc import FuncDesignerException, Diag, Eye, pWarn, scipyAbsentMsg
from copy import deepcopy

try:
    from DerApproximator import get_d1, check_d1
    DerApproximatorIsInstalled = True
except:
    DerApproximatorIsInstalled = False


try:
    import scipy
    SparseMatrixConstructor = lambda *args, **kwargs: scipy.sparse.lil_matrix(*args, **kwargs)
    from scipy import sparse
    Hstack = sparse.hstack
    from scipy.sparse import isspmatrix
except:
    scipy = None
    isspmatrix = lambda *args,  **kwargs:  False

class oofun:
    #TODO:
    #is_oovarSlice = False
    
    d = None # derivative
    input = None 
    args = ()
    #usedIn =  set()
    isAlreadyPrepared = False
    is_oovar = False
    is_linear = False
    isConstraint = False
    #isDifferentiable = True
    discrete = False
    isCostly = False
    
    stencil = 2 # used for DerApproximator
    
    #TODO: modify for cases where output can be partial
    evals = 0
    same = 0
    same_d = 0
    evals_d  = 0

    # finite-difference aproximation step
    diffInt = 1.5e-8
    maxViolation = 1e-2
    _unnamedFunNumber = 1

    _usedIn = 0
    _directlyDwasInwolved = False

    pWarn = lambda self, msg: pWarn(msg)
    
    __array_priority__ = 15# set it greater than 1 to prevent invoking numpy array __mul__ etc

    
    def __getattr__(self, attr):
        if attr != 'size': raise AttributeError
        
        # to prevent creating of several oofuns binded to same oofun.size
        r = oofun(lambda x: asarray(x).size, input = self, is_linear=True, discrete = True)
        self.size = r 
        
        return r

    """                                         Class constructor                                   """

    def __init__(self, fun, *args, **kwargs):
        assert len(args) == 0
        self.fun = fun
        
        if 'name' not in kwargs.keys():
            self.name = 'unnamed_oofun_' + str(oofun._unnamedFunNumber)
            oofun._unnamedFunNumber += 1

        for key, item in kwargs.iteritems():
            #assert key in self.__allowedFields__ # TODO: make set comparison
            setattr(self, key, item)
        if hasattr(self, 'input'):
            if type(self.input) not in [list, tuple]:
                self.input = [self.input]
            for elem in self.input:
                if isinstance(elem, oofun):
                    elem._usedIn += 1
                else:
                    pass
                    #raise FuncDesignerException('All input(s) of the oofun ' + self.name + ' have to be oofun/oovar instance(s)')
    
    def named(self, name):
        self.name = name
        return self
        
    def __repr__(self):
        return self.name
        
    # overload "a+b"
    # @checkSizes
    def __add__(self, other):
        if not isinstance(other, oofun) and not isscalar(other) and not isinstance(other, ndarray) and not isinstance(other, list):
            raise FuncDesignerException('operation oofun_add is not implemented for the type ' + str(type(other)))
            
        
            
        # TODO: check for correct sizes during f, not only f.d 
    
        def aux_d(x, y):
            if x.size == 1:
                return ones(y.size)
            elif y.size == 1:
                return Eye(x.size)
            elif x.size == y.size:
                return Eye(y.size)
            else:
                raise FuncDesignerException('for oofun summation a+b should be size(a)=size(b) or size(a)=1 or size(b)=1')        

        
        if isinstance(other, oofun):
            r = oofun(lambda x, y: x+y, input = [self, other], d = (lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x)))
        else:
            other = asarray(other)
            r = oofun(lambda a: a+other, input = self)# TODO: involve sparsity if possible!
            r.d = lambda x: aux_d(x, other)
                
# TODO: create linear field with operators +, -, *(non-oofunc), /(non-oofunc)
        if self.is_linear and (not isinstance(other, oofun) or other.is_linear): 
            r.is_linear = True
        return r
        #r = oofun(lambda point: )
    
    def __radd__(self, other):
        return self.__add__(other)
    
    # overload "-a"
    def __neg__(self): 
        return oofun(lambda a: -a, input = self, \
                     d = lambda a: -1.0 if a.size == 1 else -Eye(a.size), \
                     is_linear = True if self.is_linear else False)
        
    # overload "a-b"
    def __sub__(self, other):
        if isinstance(other, list) and type(other[0]) in (int, float):
            return self + (-asfarray(other))
            other = -asfarray(other)
        else:
            return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # overload "a/b"
    def __div__(self, other):
        if isinstance(other, oofun):
            r = oofun(lambda x, y: x/y, input = [self, other])
            def aux_dx(x, y):
                x, y, = asarray(x), asarray(y)
                if x.size != 1:
                    assert x.size == y.size or y.size == 1, 'incorrect size for oofun devision'
                r = 1.0 / y
                if x.size != 1:
                    if y.size == 1: r = r.tolist() * x.size
                    r = Diag(r)
                return r                
            def aux_dy(x, y):
                r = -x / y**2
                if y.size != 1:
                    assert x.size == y.size or x.size == 1, 'incorrect size for oofun devision'
                    r = Diag(r)
                return r
            r.d = (aux_dx, aux_dy)
        else:
            r = oofun(lambda a: a/asarray(other), input = self)# TODO: involve sparsity if possible!
            r.d = lambda x: 1.0/asarray(other) if x.size == 1 else Diag(ones(x.size)/asarray(other))
        if self.is_linear and not isinstance(other, oofun):
            r.is_linear = True
        #r.isCostly = True
        return r

    def __rdiv__(self, other):
        other = asarray(other) # TODO: sparse matrices handling!
        r = oofun(lambda x: other/x, input = self)# TODO: involve sparsity if possible!
        def d(x):
            if other.size > 1 or x.size > 1: return Diag(- other / x**2)
            else: return -other / x**2
        r.d = d
        #r.isCostly = True
        return r

    # overload "a*b"
    def __mul__(self, other):
        def aux_d(x, y):
            #assert y.size <100000
            if x.size == 1:
                return y.copy()
            elif y.size == 1:
                r = empty(x.size)
                r.fill(y)
                return Diag(r)
            elif x.size == y.size:
                return Diag(y)
            else:
                raise FuncDesignerException('for oofun multiplication a*b should be size(a)=size(b) or size(a)=1 or size(b)=1')                    
        
        if isinstance(other, oofun):
            def f(x, y):
                if x.size != 1 and y.size != 1 and x.size != y.size:
                    raise FuncDesignerException('for oofun multiplications a*b should be size(a)=size(b) or size(a)=1 or size(b)=1')
                return x*y
                
            r = oofun(f, input = [self, other])
            
            r.d = (lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x))
        else:
            r = oofun(lambda x: x*other, input = self)# TODO: involve sparsity if possible!
            r.d = lambda x: aux_d(x, asfarray(other))
        if self.is_linear and not isinstance(other, oofun):
            r.is_linear = True
        #r.isCostly = True
        return r

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        
        d_x = lambda x, y: y * x ** (y - 1) if x.size == 1 else Diag(y * x ** (y - 1))

        d_y = lambda x, y: x ** y * log(x) if y.size == 1 else Diag(x ** y * log(x))
            
        if not isinstance(other, oofun):
            f = lambda x: x ** asarray(other)
            d = lambda x: d_x(x, asarray(other))
            input = self
        else:
            f = lambda x, y: x ** y
            d = (d_x, d_y)
            input = [self, other]
        r = oofun(f, d = d, input = input)
        #r.isCostly = True
        return r

    def __rpow__(self, other):
        assert not isinstance(other, oofun)# if failed - check __pow__implementation
            
        f = lambda x: asarray(other) ** x
        #d = lambda x: Diag(asarray(other) **x * log(asarray(other)))
        def d(x):
            r = Diag(asarray(other) **x * log(asarray(other))) if x.size > 1 else asarray(other)**x * log(asarray(other))
            #raise 0
            return r
        r = oofun(f, d=d, input = self)
        #r.isCostly = True
        return r

    def __xor__(self, other):
        raise FuncDesignerException('For power of oofuncs use a**b, not a^b')
        
    def __rxor__(self, other):
        raise FuncDesignerException('For power of oofuncs use a**b, not a^b')
        
    def __getitem__(self, ind): # overload for oofun[ind]
        assert not isinstance(ind, oofun), 'slicing by oofuns is unimplemented yet'
        if isinstance(ind, dict): return self.__call__(ind) # value in a point

        if  not hasattr(self, '_slicesIndexDict'):
            self._slicesIndexDict = {}
        if ind in self._slicesIndexDict:
            return self._slicesIndexDict[ind]

        if not isinstance(ind, oofun):
            f = lambda x: x[ind]
            def d(x):
                r = zeros(x.shape)
                r[ind] = 1
                return r
        else:
            f = lambda x, _ind: x[_ind]
            def d(x, _ind):
                r = zeros(x.shape)
                r[_ind] = 1
                return r
                
        r = oofun(f, input = self, d = d, size = 1)
        # TODO: check me!
        # what about a[a.size/2:]?
        if self.is_linear and not isinstance(ind,  oofun):
            r.is_linear = True
            
            
        # TODO: edit me!
#        if self.is_oovar:
#            r.is_oovarSlice = True
        self._slicesIndexDict[ind] = r
        return r
    
    def __getslice__(self, ind1, ind2):# overload for oofun[ind1:ind2]
    
        #raise FuncDesignerException('oofun slicing is not implemented yet')
        
        assert not isinstance(ind1, oofun) and not isinstance(ind2, oofun), 'slicing by oofuns is unimplemented yet'
        f = lambda x: x[ind1:ind2]
        def d(x):
            condBigMatrix = x.size > 100 #and (ind2-ind1) > 0.25*x.size
            if condBigMatrix and scipy is None:
                self.pWarn(scipyAbsentMsg)
            if condBigMatrix and scipy is not None:
                m1 = SparseMatrixConstructor((ind2-ind1, ind1))
                m2 = Eye(ind2-ind1)
                m3 = SparseMatrixConstructor((ind2-ind1, x.size - ind2))
                r = Hstack((m1, m2, m3))
            else:
                m1 = zeros((ind2-ind1, ind1))
                m2 = eye(ind2-ind1)
                m3 = zeros((ind2-ind1, x.size - ind2))
                r = hstack((m1, m2, m3))
            return r
        r = oofun(f, input = self, d = d)
        if self.is_linear:
            r.is_linear = True
        return r
   
#    def __len__(self):
        #return self.size
        #raise FuncDesignerException('using len(obj) (where obj is oovar or oofun) is not possible (at least yet), use obj.size instead')

    def sum(self):
        r = oofun(sum, input = self)
        def d(x):
            if x.ndim > 1: raise FuncDesignerException('sum(x) is not implemented yet for arrays with ndim > 1')
            return ones(x.size) 
        r.d, r.is_linear = d, self.is_linear
        return r
    
    def prod(self):
        # TODO: consider using r.isCostly = True
        r = oofun(prod, input = self)
        def d(x):
            if x.ndim > 1: raise FuncDesignerException('prod(x) is not implemented yet for arrays with ndim > 1')
            ind_zero = where(x==0)[0].tolist()
            ind_nonzero = nonzero(x)[0].tolist()
            numOfZeros = len(ind_zero)
            r = prod(x) / x
            
            if numOfZeros >= 2: 
                r[ind_zero] = 0
            elif numOfZeros == 1:
                r[ind_zero] = prod(x[ind_nonzero])

            return r 
        r.d = d
        return r


    """                                     Handling constraints                                  """
    
    # TODO: optimize for lb-ub imposed on oovars
    
    # TODO: fix it for discrete problems like MILP, MINLP
    def __gt__(self, other): # overload for >
        if self.is_oovar and not isinstance(other, oofun):
            r = BoxBoundConstraint(self, lb = other)
        elif self.is_linear and (not isinstance(other, oofun) or other.is_linear):
            r = LinearConstraint(self-other, lb = 0)
        else:
            r = NonLinearConstraint(self - other, lb=0) # do not perform check for other == 0, copy should be returned, not self!
        #r.type = 'ineq'
        return r

    def __ge__(self, other): # overload for >=
        return self.__gt__(other)

    # TODO: fix it for discrete problems like MILP
    def __lt__(self, other): # overload for <
        # TODO:
        #(self.is_oovar or self.is_oovarSlice)
        if self.is_oovar and not isinstance(other, oofun):
            r = BoxBoundConstraint(self, ub = other)
        elif self.is_linear and (not isinstance(other, oofun) or other.is_linear):
            r = LinearConstraint(self-other, ub = 0)
        else:
            r = NonLinearConstraint(self - other, ub = 0) # do not perform check for other == 0, copy should be returned, not self!
        #r.type = 'ineq'
        return r            


    def __le__(self, other): # overload for <=
        return self.__lt__(other)
    
    def eq(self, other):
        if self.is_oovar and not isinstance(other, oofun):
            raise FuncDesignerException('Constraints like this: "myOOVar = <some value>" are not implemented yet and are not recommended; for openopt use optVars / fixedVars instead')
        if self.is_linear and (not isinstance(other, oofun) or other.is_linear):
            r = LinearConstraint(self-other, ub = 0, lb = 0)
        else:
            r = NonLinearConstraint(self - other, ub = 0, lb = 0) # do not perform check for other == 0, copy should be returned, not self!
        #r.type = 'ineq'
        return r  
            
#    def __eq__(self, other):
#        raise FuncDesignerException('Operator == is unavailable, at least yet')

#    def __eq__(self, other):
#        r = self - other
#        r.isConstraint = True
#        r.type = 'eq'
#        return r


    """                                             getInput                                              """
    def _getInput(self, x):
        if not type(self.input) in (list, tuple):
            self.input = [self.input]
        r = []
        self.inputOOVarTotalLength = 0
        for item in self.input:
            if isinstance(item, oofun): 
                rr = atleast_1d(item(x))
            else:
                rr = atleast_1d(item)
            r.append(rr)
            self.inputOOVarTotalLength += rr.size
        return tuple(r)

    """                                                getDep                                             """
    def _getDep(self):
        if hasattr(self, 'dep'):
            return self.dep
        elif self.input is None:
            self.dep = None
        else:
            r = set([])
            #r.fill(False)
            if not type(self.input) in (list, tuple):
                self.input = [self.input]
            for oofunInstance in self.input:
                if not isinstance(oofunInstance, oofun): continue
                # TODO: checkme!
                
                #!!!!!!!!!!!!!!!!!!! TODO: HANDLE FIXED OOFUNCS
                #if oofunInstance.fixed: continue
                
                if oofunInstance.is_oovar:
                    r.add(oofunInstance)
                    continue
                
                tmp = oofunInstance._getDep()
                
                if type(tmp) != set:
                    raise FuncDesignerException('unknown type of oofun or oovar dependence')
                r.update(tmp)
            self.dep = r    
        return self.dep


    """                                                getFunc                                             """
    def _getFunc(self, x):
        # TODO: get rid of hasattr(self, 'fixed')
        #if self.isAlreadyPrepared:
        
        #!!!!!!!!!!!!!!!!!!! TODO: HANDLE FIXED OOFUNCS
        
#        if self.fixed and hasattr(self, 'f_key_prev'):
#            # TODO: remove duplicated code from the func
#            if isinstance(self.f_val_prev, ndarray) or isinstance(self.f_val_prev, float):
#                return atleast_1d(copy(self.f_val_prev))
#            else:
#                return deepcopy(self.f_val_prev)
            
        dep = self._getDep()
        cond_same_point = self.isCostly and hasattr(self, 'f_key_prev') and all([array_equal(x[elem], self.f_key_prev[elem.name]) for elem in dep])
        #cond_same_point = hasattr(self, 'f_key_prev') and all([array_equal(x[elem], self.f_key_prev[elem.name]) for elem in dep])
        if cond_same_point:
            self.same += 1
            return deepcopy(self.f_val_prev)
            
        self.evals += 1
        
        if type(self.args) != tuple:
            self.args = (self.args, )
        Input = self._getInput(x)
        if self.args != ():
            Input += self.args
        tmp = self.fun(*Input)
        if isinstance(tmp, list) or isinstance(tmp, tuple):
            tmp = hstack(tmp)
        
        #self.outputTotalLength = ([asarray(elem).size for elem in self.fun(*Input)])#self.f_val_prev.size # TODO: omit reassigning
        
        # TODO: simplify it
        if self.isCostly:
            self.f_val_prev = tmp
            self.f_key_prev = {}
            for elem in dep:
                self.f_key_prev[elem.name] = copy(x[elem])
                
            if isinstance(self.f_val_prev, ndarray) or isscalar(self.f_val_prev):
                return atleast_1d(copy(self.f_val_prev))
            else:
                return deepcopy(self.f_val_prev)
        else:
            return tmp


    """                                                getFunc                                             """
    __call__ = lambda self, *args: self._getFunc(*args)


    """                                              derivatives                                           """
    def D(self, x, Vars=None, fixedVars = None, resultKeysType = 'vars', asSparse = False):
        # resultKeysType doesn't matter for the case isinstance(Vars, oovar)
        if Vars is not None and fixedVars is not None:
            raise FuncDesignerException('No more than one argument from "Vars" and "fixedVars" is allowed for the function')
        assert type(Vars) != ndarray and type(fixedVars) != ndarray
        self._directlyDwasInwolved = True
        if not hasattr(self, '_prev_D_Vars_key'):
            self._prev_D_Vars_key = Vars if (Vars is None or (isinstance(Vars, oofun) and Vars.is_oovar) ) else set([v.name for v in Vars])
            self._prev_D_fixedVars_key = fixedVars if (fixedVars is None or (isinstance(fixedVars, oofun) and fixedVars.is_oovar) ) \
            else set([v.name for v in fixedVars])
        sameDerivativeVariables = True if (self._prev_D_Vars_key == Vars and self._prev_D_fixedVars_key == fixedVars) else False
        
        #TODO: remove cloned code
        if Vars is not None:
            if type(Vars) in [list, tuple]:
                Vars = set(Vars)
            elif isinstance(Vars, oofun):
                if not Vars.is_oovar:
                    raise FuncDesignerException('argument Vars is expected as oovar or python list/tuple of oovar instances')
                Vars = set([Vars])
        if fixedVars is not None:
            if type(fixedVars) in [list, tuple]:
                fixedVars = set(fixedVars)
            elif isinstance(fixedVars, oofun):
                if not fixedVars.is_oovar:
                    raise FuncDesignerException('argument fixedVars is expected as oovar or python list/tuple of oovar instances')
                fixedVars = set([fixedVars])
        r = self._D(x, Vars, fixedVars, sameDerivativeVariables = sameDerivativeVariables, asSparse = asSparse)
        if isinstance(Vars, oofun):
            if Vars.is_oovar:
                return Vars(r)
            else: 
                # TODO: handle it with input of type list/tuple/etc as well
                raise FuncDesignerException('Cannot perform differentiation by non-oovar input')
        else:
            if resultKeysType == 'names':
                return r
            elif resultKeysType == 'vars':
                rr = {}
                tmpDict = Vars if Vars is not None else x
                for oov, val in x.items():
                    if oov.name not in r.keys() or (fixedVars is not None and oov in fixedVars):
                        continue
                    rr[oov] = r[oov.name]
                return rr
            else:
                raise FuncDesignerException('Incorrect argument resultKeysType, should be "vars" or "names"')
            
            
    def _D(self, x, Vars=None, fixedVars = None, sameDerivativeVariables=True, asSparse = 'auto'):
        if self.is_oovar: 
            return {} if fixedVars is not None and self in fixedVars else {self.name:Eye(self(x).size) if asSparse is not False else eye(self(x).size)}
        if self.discrete: raise FuncDesignerException('The oofun or oovar instance has been declared as discrete, no derivative is available')
        if Vars is not None and fixedVars is not None:
            raise FuncDesignerException('No more than one parameter from Vars and fixedVars is allowed')
      
        dep = self._getDep()
        ##########################
        
        # TODO: optimize it. Omit it for simple cases.
        isTransmit = self._usedIn == 1 # Exactly 1! not 0, 2, ,3, 4, etc
        involveStore = (not isTransmit) or self._directlyDwasInwolved
        #cond_same_point = False
        cond_same_point = involveStore and sameDerivativeVariables and \
        hasattr(self, 'd_key_prev') and all([array_equal(x[elem], self.d_key_prev[elem.name]) for elem in dep])
        
        if cond_same_point:
            self.same_d += 1
            return deepcopy(self.d_val_prev)
        else:
            self.evals_d += 1
        
        derivativeSelf = self._getDerivativeSelf(x, Vars, fixedVars)
        
        r = Derivative()
        Keys = set()
        ac = -1
        for i, inp in enumerate(self.input):
            if not isinstance(inp, oofun): continue
            if inp.discrete: continue
            if fixedVars is not None and inp.is_oovar and inp in fixedVars: continue
            #!!!!!!!!! TODO: handle fixed cases properly!!!!!!!!!!!!
            #if hasattr(inp, 'fixed') and inp.fixed: continue

            # TODO: check for unique oovar names!
            if inp.is_oovar: 
                if (Vars is not None and inp not in Vars) or (fixedVars is not None and inp in fixedVars):
                    continue                
                ac += 1
                tmp = derivativeSelf[ac]
                
                if (tmp.ndim <= 1 or min(tmp.shape) == 1) and not isspmatrix(tmp):
                    tmp = tmp.flatten()

                #Key = inp
                Key  = inp.name
                if Key in Keys:
                    if prod(tmp.shape) <= prod(r[Key].shape): 
                        r[Key] += tmp
                    else:
                        r[Key] = r[Key] + tmp
                else:
                    r[Key] = tmp
                    Keys.add(Key)
            else:
                ac += 1
                
                # asSparse should be 'autoselect' here, not the value taken from the function arguments!
                elem_d = inp._D(x, Vars, fixedVars, sameDerivativeVariables, asSparse = 'auto') 
                
                for key in elem_d.keys():
                    if prod(derivativeSelf[ac].shape) == 1 or prod(elem_d[key].shape) == 1:
                        rr = derivativeSelf[ac] * elem_d[key]
                    else:
                        t1, t2 = self._considerSparse(derivativeSelf[ac], elem_d[key])
                        cond_2 = t1.ndim > 1 or t2.ndim > 1
                        if cond_2:
                            # warning! t1,t2 can be sparse matrices, so I don't use t = atleast_2d(t) directly
                            #if t1.ndim < 2: t1 = atleast_2d(t1) 
                            if t2.ndim < 2: 
                                assert t1.ndim > 1, 'error in FuncDesigner kernel, inform developers'
                                if t1.shape[1] != t2.shape[0]:
                                    t2 = t2.reshape(1, -1)
                                #t2 = atleast_2d(t2)
                        else:
                            # hence these are ndarrays
                            if self(x).size > 1:
                                t1 = t1.reshape(-1, 1)
                                t2 = t2.reshape(1, -1)
                            else:
                                t1 = t1.flatten()
                                t2 = t2.flatten()
                        
                        if not (isinstance(t1,  ndarray) and isinstance(t2,  ndarray)):
                            if scipy is None:
                                self.pWarn(scipyAbsentMsg)
                                rr = atleast_1d(dot(t1, t2))
                            else:
                                t1 = scipy.sparse.csr_matrix(t1)
                                t2 = scipy.sparse.csc_matrix(t2)
                                if t2.shape[0] != t1.shape[1]:
                                    if t2.shape[1] == t1.shape[1]:
                                        t2 = t2.T
                                    else:
                                        raise FuncDesignerException('incorrect shape in FuncDesigner function _D(), inform developers about the bug')
                                rr = t1._mul_sparse_matrix(t2)
                                if asSparse is False:
                                    rr = rr.toarray() 
                        else:
                            rr = atleast_1d(dot(t1, t2))
                        #assert rr.size != 178784

#                    if min(rr.shape) == 1 and isinstance(rr, ndarray): 
#                        rr = rr.flatten() # TODO: check it and mb remove
# TODO: implement it instead of the code below
                    
                    
                    if min(rr.shape) == 1: 
                        if not isinstance(rr, ndarray): 
                            rr = rr.toarray()
                        rr = rr.flatten() # TODO: check it and mb remove
                    if key in Keys:
                        if isinstance(r[key], ndarray) and not isinstance(rr, ndarray): # i.e. rr is sparse matrix
                            rr = rr.toarray() # I guess r[key] will hardly be all-zeros
                        elif not isinstance(r[key], ndarray) and isinstance(rr, ndarray): # i.e. r[key] is sparse matrix
                            r[key] = r[key].toarray()
                        if rr.size == r[key].size and isinstance(rr, ndarray): 
                            r[key] += rr
                        else: 
                            r[key] = r[key] + rr
                    else:
                        r[key] = rr
                        Keys.add(key)
                    assert r[key].dtype != object

        dp = {}
        for key, value in r.items():
            dp[key] = value.copy()
        self.d_val_prev = dp
        
        self.d_key_prev = {}
        for elem in dep:
            self.d_key_prev[elem.name] = copy(x[elem])
        
        return r

    # TODO: handle 2**15 & 0.25 as parameters
    def _considerSparse(self, t1, t2):  
        #assert self.name != 'unnamed_oofun_81'
        if prod(t1.shape) * prod(t2.shape) > 2**15 and   (isinstance(t1, ndarray) and t1.nonzero()[0].size < 0.25*t1.size) or \
        (isinstance(t2, ndarray) and t2.nonzero()[0].size < 0.25*t2.size):
            if scipy is None: 
                self.pWarn(scipyAbsentMsg)
                return t1,  t2
            t1 = scipy.sparse.csr_matrix(t1)
            if t1.shape[0] == 1: t1 = t1.T
            t2 = scipy.sparse.csc_matrix(t2)
        return t1,  t2

    def _getDerivativeSelf(self, x, Vars,  fixedVars):
        Input = self._getInput(x)
        if hasattr(self, 'size') and isscalar(self.size): nOutput = self.size
        else: nOutput = self(x).size 
        hasUserSuppliedDerivative = hasattr(self, 'd') and self.d is not None
        if hasUserSuppliedDerivative:
            derivativeSelf = []
            if type(self.d) == tuple:
                if len(self.d) != len(self.input):
                   raise FuncDesignerException('oofun error: num(derivatives) not equal to neither 1 nor num(inputs)')
                   
                indForDiff = []
                for i, deriv in enumerate(self.d):
                    inp = self.input[i]
                    if not isinstance(inp, oofun) or inp.discrete: 
                        #if deriv is not None: 
                            #raise FuncDesignerException('For an oofun with some input oofuns declared as discrete you have to set oofun.d[i] = None')
                        continue
                    
                    #!!!!!!!!! TODO: handle fixed cases properly!!!!!!!!!!!!
                    #if hasattr(inp, 'fixed') and inp.fixed: continue
                    if inp.is_oovar and ((Vars is not None and inp not in Vars) or (fixedVars is not None and inp in fixedVars)):
                        continue
                        
                    # TODO: implement it properly + related changes in _D()
                    #if not inp.is_oovar and self.theseAreFixed(set(inp._getDep())): continue
                    
                    if deriv is None:
                        if not DerApproximatorIsInstalled:
                            raise FuncDesignerException('To perform gradients check you should have DerApproximator installed, see http://openopt.org/DerApproximator')
                        derivativeSelf.append(get_d1(self.fun, Input, diffInt=self.diffInt, stencil = self.stencil, args=self.args, varForDifferentiation = i, pointVal = self._getFunc(x)))
                    else:
                        # !!!!!!!!!!!!!! TODO: add check for user-supplied derivative shape
                        tmp = deriv(*Input)
                        if isscalar(tmp) or type(tmp) in (ndarray, tuple, list): # i.e. not a scipy.sparse matrix
                            tmp = atleast_2d(tmp)
                            if tmp.shape[0] != nOutput: 
                                # TODO: add debug msg
                                if tmp.shape[1] != nOutput: raise FuncDesignerException('error in getDerivativeSelf()')
                                tmp = tmp.T
                        derivativeSelf.append(tmp)
            else:
                tmp = self.d(*Input)
                if isscalar(tmp) or type(tmp) in (ndarray, tuple, list): # i.e. not a scipy.sparse matrix
                    tmp = atleast_2d(tmp)
                    if tmp.shape[0] != nOutput: 
                        # TODO: add debug msg
                        if tmp.shape[1] != nOutput: raise FuncDesignerException('error in getDerivativeSelf()')
                        tmp = tmp.T
                        
                   
                ac = 0
                
                if not isinstance(tmp, ndarray):
                    csr_tmp = tmp.tocsr()
                for i, inp in enumerate(Input):
                    if isinstance(tmp, ndarray):
                        TMP = tmp[:, ac:ac+inp.size]
                    else: # scipy.sparse matrix
                        TMP = csr_tmp[:, ac:ac+inp.size]
                    ac += inp.size
                    if self.input[i].discrete: continue
                    #!!!!!!!!! TODO: handle fixed cases properly!!!!!!!!!!!!
                    #if hasattr(self.input[i], 'fixed') and self.input[i].fixed: continue 
                    if self.input[i].is_oovar and ((Vars is not None and self.input[i] not in Vars) or (fixedVars is not None and self.input[i] in fixedVars)):
                        continue                                    
                    
                    #if Input[i].size == 1: TMP = TMP.flatten()
                    if isinstance(TMP, ndarray) and min(TMP.shape) == 1: TMP = TMP.flatten()
                    derivativeSelf.append(TMP)
                    
            # TODO: is it required?
#                if not hasattr(self, 'outputTotalLength'): self._getFunc(x)
#                
#                if derivativeSelf.shape != (self.outputTotalLength, self.inputTotalLength):
#                    s = 'incorrect shape for user-supplied derivative of oofun '+self.name+': '
#                    s += '(%d, %d) expected, (%d, %d) obtained' % (self.outputTotalLength, self.inputTotalLength,  derivativeSelf.shape[0], derivativeSelf.shape[1])
#                    raise FuncDesignerException(s)
        else:
            if Vars is not None or fixedVars is not None: raise ("sorry, custom oofun derivatives don't work with Vars/fixedVars arguments yet")
            if not DerApproximatorIsInstalled:
                raise FuncDesignerException('To perform gradients check you should have DerApproximator installed, see http://openopt.org/DerApproximator')
            derivativeSelf = get_d1(self.fun, Input, diffInt=self.diffInt, stencil = self.stencil, args=self.args, pointVal = self._getFunc(x))
        
        # TODO: it should be handled in a better way, with personal derivatives for named inputs
        if isinstance(derivativeSelf, ndarray):
            assert len(self.input) == 1, 'error in package engine, please inform developers'
            derivativeSelf = [derivativeSelf]
        #print 'derivativeSelf:', derivativeSelf
        #assert derivativeSelf[0].dtype == float
        return derivativeSelf

    def D2(self, x):
        raise FuncDesignerException('2nd derivatives for obj-funcs are not implemented yet')

    def check_d1(self, point):
        if self.d is None:
            print('Error: no user-provided derivative(s) for oofun ' + self.name + ' are attached')
            return # TODO: return non-void result
        separator = 75 * '*'
        print(separator)
        assert type(self.d) != list
        val = self._getFunc(point)
        input = self._getInput(point)
        ds= self._getDerivativeSelf(point, Vars=None,  fixedVars=None)
        print(self.name + ': checking user-supplied gradient')
        print('according to:')
        print('    diffInt = ' + str(self.diffInt)) # TODO: ADD other parameters: allowed epsilon, maxDiffLines etc
        print('    |1 - info_user/info_numerical| < maxViolation = '+ str(self.maxViolation))        
        j = -1
        for i in xrange(len(self.input)):#var in Vars:
            var = self.input[i]
            if len(self.input) > 1: print('by input variable number ' + str(i) + ':')
            if isinstance(self.d, tuple) and self.d[i] is None:
                print('user-provided derivative for input number ' + str(i) + ' is absent, skipping the one;')
                print(separator)
                continue
            if not isinstance(self.input[i], oofun):
                print('input number ' + str(i) + ' is not oofun instance, skipping the one;')
                print(separator)
                continue
            j += 1
            check_d1(lambda *args: self.fun(*args), ds[j], input, \
                 func_name=self.name, diffInt=self.diffInt, pointVal = val, args=self.args, \
                 stencil = max((2, self.stencil)), maxViolation=self.maxViolation, varForCheck = i)


class BooleanOOFun(oofun):
    _unnamedBooleanOOFunNumber = 1
    discrete = True
    # an oofun that returns True/False
    def __init__(self, *args, **kwargs):
        self.name = 'unnamed_boolean_oofun_' + str(BooleanOOFun._unnamedBooleanOOFunNumber)
        BooleanOOFun._unnamedBooleanOOFunNumber += 1

class BaseFDConstraint(BooleanOOFun):
    isConstraint = True
    contol = 1e-6
    def __init__(self, oofun_Involved, *args, **kwargs):
        BooleanOOFun.__init__(self, *args, **kwargs)
        #oofun.__init__(self, lambda x: oofun_Involved(x), input = oofun_Involved)
        if len(args) != 0:
            raise FuncDesignerException('No args are allowed for FuncDesigner constraint constructor, only some kwargs')
            
        # TODO: replace self.oofun by self.engine
        self.oofun = oofun_Involved
        
        if 'contol' in kwargs:
            self.contol = kwargs['contol']

class SmoothFDConstraint(BaseFDConstraint):
    isBBC = False
    
    def __call__(self, point, contol=0):
        #if contol is None: contol = 0#self.contol
        val = self.oofun(point)
        if any(atleast_1d(self.lb-val)>contol):
            return False
        elif any(atleast_1d(val-self.ub)>contol):
            return False
        return True
        
    __getitem__ = lambda self, point: self.__call__(point, self.contol)
        
    def __init__(self, *args, **kwargs):
        BaseFDConstraint.__init__(self, *args, **kwargs)
        self.lb, self.ub = -inf, inf
        for key in kwargs.keys():
            if key in ['lb', 'ub']:
                setattr(self, key, asarray(kwargs[key]))
            else:
                raise FuncDesignerException('Unexpected key in FuncDesigner constraint constructor kwargs')
    

class NonLinearConstraint(SmoothFDConstraint):
    def __init__(self, *args, **kwargs):
        SmoothFDConstraint.__init__(self, *args, **kwargs)
        
        
class BoxBoundConstraint(SmoothFDConstraint):
    isBBC = True
    def __init__(self, *args, **kwargs):
        SmoothFDConstraint.__init__(self, *args, **kwargs)
        
class LinearConstraint(SmoothFDConstraint):
    def __init__(self, *args, **kwargs):
        SmoothFDConstraint.__init__(self, *args, **kwargs)

class Derivative(dict):
    def __init__(self):
        pass


def ooFun(*args, **kwargs):
    r = oofun(*args, **kwargs)
    r.isCostly = True
    return r


#
#class ooconstraint(oofun):
#    def __init__(self, oofun_instance):
#        self.oofun = oofun_instance

#class oolin(oofun):
#    def __init__(self, C, d=0, *args, **kwargs):
#        # returns Cx + d
#        # TODO: handle FIXED variables here
#        mtx = atleast_2d(array(C, float))
#        d = array(d, float)
#        self.mult, self.add = mtx, d
#        
#        # TODO: use p.err instead assert
#        assert d.ndim <= 1, 'passing d with ndim>1 into oolin Cx+d is forbidden'
#        if d.size != mtx.shape[0]:
#            if d.size == 1: FuncDesignerException('Currently for Cx+d using d with size 1 is forbidden for C.shape[0]>1 for the sake of more safety and for openopt users code to be clearer')
#        
#        ind_zero = where(all(mtx==0, 0))[0]
#        def oolin_objFun(*x):
#            if len(x) == 1:
#                x = x[0]
#            X = asfarray(x).ravel()
#            X[ind_zero] = 0
#            r = dot(mtx, X) + d # case c = 0 or all-zeros yields insufficient additional calculations, so "if c~=0" can be omitted
#            return r
#        oofun.__init__(self, oolin_objFun, *args, **kwargs)
#        
#        #derivative:
#        self.d = lambda *x: mtx.copy()
