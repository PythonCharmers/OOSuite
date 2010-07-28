# created by Dmitrey

from numpy import inf, asfarray, copy, all, any, empty, atleast_2d, zeros, dot, asarray, atleast_1d, empty, ones, ndarray, \
where, array, nan, ix_, vstack, eye, array_equal, isscalar, diag, log, hstack, sum, prod, nonzero, isnan, asscalar
from numpy.linalg import norm
from misc import FuncDesignerException, Diag, Eye, pWarn, scipyAbsentMsg, scipyInstalled
from copy import deepcopy
from ooPoint import ooPoint

#def Copy(elem, useSparse):
#    # useSparse can be changed while saved data is involved
#    if useSparse is False and isspmatrix(elem):
#        return elem.copy().toarray()
#    elif hasattr(elem, 'copy'): return elem.copy()
#    else: return copy(elem)
Copy = lambda arg: asscalar(arg) if type(arg)==ndarray and arg.size == 1 else arg.copy() if hasattr(arg, 'copy') else copy(arg)

try:
    from DerApproximator import get_d1, check_d1
    DerApproximatorIsInstalled = True
except:
    DerApproximatorIsInstalled = False


try:
    import scipy
    SparseMatrixConstructor = lambda *args, **kwargs: scipy.sparse.lil_matrix(*args, **kwargs)
    from scipy import sparse
    from scipy.sparse import hstack as HstackSP
    def Hstack(Tuple):
        #elems = asarray(Tuple, dtype=object)
        ind = where([prod(elem.shape)!=0 for elem in Tuple])[0].tolist()
        elems = [Tuple[i] for i in ind]
        # [elem if prod(elem.shape)!=0 for elem in Tuple]
        return HstackSP(elems) if any([isspmatrix(elem) for elem in elems]) else hstack(elems)
        
    from scipy.sparse import isspmatrix
except:
    scipy = None
    isspmatrix = lambda *args,  **kwargs:  False

class oofun:
    #TODO:
    #is_oovarSlice = False
    
    d = None # derivative
    input = None#[] 
    #usedIn =  set()
    is_oovar = False
    is_linear = False
    isConstraint = False
    #isDifferentiable = True
    discrete = False
    isCostly = False
    
    stencil = 3 # used for DerApproximator
    
    #TODO: modify for cases where output can be partial
    evals = 0
    same = 0
    same_d = 0
    evals_d  = 0

    # finite-difference aproximation step
    diffInt = 1.5e-8
    maxViolation = 1e-2
    _unnamedFunNumber = 1
    _lastDiffVarsID = 0

    _usedIn = 0
    _level = 0
    #_directlyDwasInwolved = False
    _id = 0
    _BroadCastID = 0
    _broadcast_id = 0
    _point_id = 0
    _point_id1 = 0
    #_c = 0.0

    pWarn = lambda self, msg: pWarn(msg)
    def disp(self, msg): 
        print(msg)
    
    __array_priority__ = 15# set it greater than 1 to prevent invoking numpy array __mul__ etc

    
    def __getattr__(self, attr):
        if attr != 'size': raise AttributeError
        
        # to prevent creating of several oofuns binded to same oofun.size
        r = oofun(lambda x: asarray(x).size, self, is_linear=True, discrete = True)
        self.size = r 

        return r

    """                                         Class constructor                                   """

    def __init__(self, fun, input=None, *args, **kwargs):
    #def __init__(self, fun, input=[], *args, **kwargs):
        assert len(args) == 0 #and input is not None
        self.fun, self.input = fun, input
        
        #self._broadcast_id = 0
        self._id = oofun._id
        self.attachedConstraints = set()
        self.args = ()
        oofun._id += 1 # CHECK: it should be int32! Other types cannot be has keys!
        
        if 'name' not in kwargs.keys():
            self.name = 'unnamed_oofun_' + str(oofun._unnamedFunNumber)
            oofun._unnamedFunNumber += 1

        for key, item in kwargs.iteritems():
            #print key
            #assert key in self.__allowedFields__ # TODO: make set comparison
            setattr(self, key, item)
            
        if input is not None:
            if type(self.input) not in [list, tuple]:
                self.input = [self.input]
            levels = [0]
            for elem in self.input: # if a
                if isinstance(elem, oofun):
                    elem._usedIn += 1
                    levels.append(elem._level)
            self._level = max(levels)+1

    def __hash__(self):
        return self._id
    
    def named(self, name):
        s = """The function "named" is deprecated and will be removed in future FuncDesigner versions, 
        instead of my_oofun.named('my name')  you should use my_oofun('my name') or my_oofun(name='my name')"""
        self.pWarn(s)
        self.name = name
        return self
        
    def attach(self, *args,  **kwargs):
        if len(kwargs) != 0:
            raise FuncDesignerException('keyword arguments are not implemented for FuncDesigner function "attach"')
        for arg in args:
            if not isinstance(arg, BaseFDConstraint):
                raise FuncDesignerException('the FD function "attach" currently expects only constraints')
            self.attachedConstraints.add(arg)
        return self
    
   
#    def _get_attached_constraints(self):
#        return self.attachedConstraints
        
        # recursively!
#        r = self.attachedConstraints
#        for inp in self.input:
#            if not isinstance(inp, oofun): continue
#            r.update(inp.attachedConstraints)
#        return r
        
    def __repr__(self):
        return self.name
        
    # overload "a+b"
    # @checkSizes
    def __add__(self, other):
        if not isinstance(other, (oofun, list, ndarray, tuple)) and not isscalar(other):
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
            r = oofun(lambda x, y: x+y, [self, other], d = (lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x)))
        else:
            other = array(other, 'float')
#            if other.size == 1:
#                #r = oofun(lambda *ARGS, **KWARGS: None, input = self.input)
#                r = oofun(lambda a: a+other, self)
#                #r._D = lambda *args,  **kwargs: self._D(*args,  **kwargs)
#                r.d = lambda x: aux_d(x, other)
#                #assert len(r._getDep())>0
#                #r._c = self._c + other
#            else:
            r = oofun(lambda a: a+other, self)
            r.d = lambda x: aux_d(x, other)
            r._getFuncCalcEngine = lambda *args,  **kwargs: self._getFuncCalcEngine(*args,  **kwargs) + other
            
            # TODO: get rid of "not self.is_oovar"
            if (other.size == 1 or ('size' in self.__dict__ and self.size == other.size)) and not self.is_oovar: 
                r._D = lambda *args,  **kwargs: self._D(*args,  **kwargs) 
                
# TODO: create linear field with operators +, -, *(non-oofunc), /(non-oofunc)
        if self.is_linear and (not isinstance(other, oofun) or other.is_linear): 
            r.is_linear = True
        return r
        #r = oofun(lambda point: )
    
    def __radd__(self, other):
        return self.__add__(other)
    
    # overload "-a"
    def __neg__(self): 
        r = oofun(lambda a: -a, self, \
                     d = lambda a: -Eye(a.size), \
                     is_linear = True if self.is_linear else False)
        r._getFuncCalcEngine = lambda *args,  **kwargs: -self._getFuncCalcEngine(*args,  **kwargs)
        def _D(*args, **kwargs):
            r = self._D(*args, **kwargs)
            return dict([(key, -value) for key, value in r.items()])
        r._D = _D
        return r
        
    # overload "a-b"
    def __sub__(self, other):
        if isinstance(other, list) and type(other[0]) in (int, float):
            return self + (-array(other, 'float'))
        else:
            return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # overload "a/b"
    def __div__(self, other):
        if isinstance(other, oofun):
            r = oofun(lambda x, y: x/y, [self, other])
            def aux_dx(x, y):
                x, y, = asfarray(x), asfarray(y) # TODO: get rid of it
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
            other = array(other,'float')
            r = oofun(lambda a: a/other, self)# TODO: involve sparsity if possible!
            r._getFuncCalcEngine = lambda *args,  **kwargs: self._getFuncCalcEngine(*args,  **kwargs) / other
            r.d = lambda x: 1.0/asfarray(other) if x.size == 1 else Diag(ones(x.size)/other)
#            if other.size == 1 or 'size' in self.__dict__ and self.size in (1, other.size):
            if not self.is_oovar and other.size == 1:
                r._D = lambda *args, **kwargs: dict([(key, value/other) for key, value in self._D(*args, **kwargs).items()])
                def _d(*args, **kwargs):
                    raise FuncDesignerException('bug in FD kernel, inform developers')
                r.d = _d
            
        if self.is_linear and not isinstance(other, oofun):
            r.is_linear = True
        #r.isCostly = True
        return r

    def __rdiv__(self, other):
        other = asfarray(other) # TODO: sparse matrices handling!
        r = oofun(lambda x: other/x, self)# TODO: involve sparsity if possible!
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
                
            r = oofun(f, [self, other])
            
            r.d = (lambda x, y: aux_d(x, y), lambda x, y: aux_d(y, x))
        else:
            other = array(other, 'float')
            r = oofun(lambda x: x*other, self)# TODO: involve sparsity if possible!
            r._getFuncCalcEngine = lambda *args,  **kwargs: other * self._getFuncCalcEngine(*args,  **kwargs)
            r.d = lambda x: aux_d(x, other)
            if not self.is_oovar and other.size == 1:
                r._D = lambda *args, **kwargs: dict([(key, other*value) for key, value in self._D(*args, **kwargs).items()])
                def _d(*args, **kwargs):
                    raise FuncDesignerException('bug in FD kernel, inform developers')
                r.d = _d
            #else:
                #r.d = lambda x: aux_d(x, asfarray(other))
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
            f = lambda x: x ** array(other, 'float')
            d = lambda x: d_x(x, array(other, 'float'))
            input = self
        else:
            f = lambda x, y: x ** y
            d = (d_x, d_y)
            input = [self, other]
        r = oofun(f, input, d = d)
        if isinstance(other, oofun) or not isinstance(other, int): r.attach((self>0)('pow_domain_%d'%r._id, tol=-1e-7)) # TODO: if "other" is fixed oofun with integer value - omit this
        r.isCostly = True
        return r

    def __rpow__(self, other):
        assert not isinstance(other, oofun)# if failed - check __pow__implementation
            
        f = lambda x: asfarray(other) ** x
        #d = lambda x: Diag(asarray(other) **x * log(asarray(other)))
        def d(x):
            r = Diag(asfarray(other) **x * log(asfarray(other))) if x.size > 1 else asfarray(other)**x * log(asfarray(other))
            return r
        r = oofun(f, self, d=d)
        r.isCostly = True
        if isinstance(other, oofun): r.attach((other>0)('rpow_domain_%d'%r._id,  tol=-1e-7)) # TODO: if "other" is fixed oofun with integer value - omit this
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
                condBigMatrix = x.size > 100 
                if condBigMatrix and scipyInstalled:
                    r = SparseMatrixConstructor((1, x.shape[0]))
                    r[0, ind] = 1.0
                else: 
                    if not scipyInstalled: self.pWarn(scipyAbsentMsg)
                    r = zeros(x.shape)
                    r[ind] = 1
                return r
        else: # NOT IMPLEMENTED PROPERLY YET
            self.pWarn('Slicing by oofuns IS NOT IMPLEMENTED PROPERLY YET')
            f = lambda x, _ind: x[_ind]
            def d(x, _ind):
                r = zeros(x.shape)
                r[_ind] = 1
                return r
                
        r = oofun(f, self, d = d, size = 1)
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
            if condBigMatrix and not scipyInstalled:
                self.pWarn(scipyAbsentMsg)
            if condBigMatrix and scipyInstalled:
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
        r = oofun(f, self, d = d)
        if self.is_linear:
            r.is_linear = True
        return r
   
    #def __len__(self):
        #return self.size
        #raise FuncDesignerException('using len(obj) (where obj is oovar or oofun) is not possible (at least yet), use obj.size instead')

    def sum(self):
        r = oofun(sum, self)
        def d(x):
            if x.ndim > 1: raise FuncDesignerException('sum(x) is not implemented yet for arrays with ndim > 1')
            return ones(x.size) 
        r.d, r.is_linear = d, self.is_linear
        return r
    
    def prod(self):
        # TODO: consider using r.isCostly = True
        r = oofun(prod, self)
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
            r = LinearConstraint(self-other, lb = 0.0)
        else:
            r = NonLinearConstraint(self - other, lb=0.0) # do not perform check for other == 0, copy should be returned, not self!
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
            r = LinearConstraint(self-other, ub = 0.0)
        else:
            r = NonLinearConstraint(self - other, ub = 0.0) # do not perform check for other == 0, copy should be returned, not self!
        return r            


    def __le__(self, other): # overload for <=
        return self.__lt__(other)
    
    def __eq__(self, other):
        return self.eq(other)
    
    def eq(self, other):
        if other in (None, (), []): return False
        if self.is_oovar and not isinstance(other, oofun):
            raise FuncDesignerException('Constraints like this: "myOOVar = <some value>" are not implemented yet and are not recommended; for openopt use optVars / fixedVars instead')
        if self.is_linear and (not isinstance(other, oofun) or other.is_linear):
            r = LinearConstraint(self-other, ub = 0.0, lb = 0.0)
        else:
            r = NonLinearConstraint(self - other, ub = 0.0, lb = 0.0) # do not perform check for other == 0, copy should be returned, not self!
        return r  
            


    """                                             getInput                                              """
    def _getInput(self, x):
#        self.inputOOVarTotalLength = 0
        return tuple([atleast_1d(item._getFuncCalcEngine(x)) if isinstance(item, oofun) else asarray(item) for item in self.input])

    """                                                getDep                                             """
    def _getDep(self):
        # returns Python set of oovars it depends on
        if hasattr(self, 'dep'):
            return self.dep
        elif self.input is None:
            self.dep = None
        else:
            if type(self.input) not in (list, tuple):
                self.input = [self.input]
            #OLD
#            r = set()
#            for oofunInstance in self.input:
#                if not isinstance(oofunInstance, oofun): continue
#                if oofunInstance.is_oovar:
#                    r.add(oofunInstance)
#                    continue
#                tmp = oofunInstance._getDep()
#                if tmp is None: continue
#                r.update(tmp)
#            self.dep = r    
            # / OLD
            
            # NEW
            r_oovars = []
            r_oofuns = []
            for oofunInstance in self.input:
                if not isinstance(oofunInstance, oofun): continue
                if oofunInstance.is_oovar:
                    r_oovars.append(oofunInstance)
                    continue
                tmp = oofunInstance._getDep()
                if tmp is None or len(tmp)==0: continue # TODO: remove None, use [] instead
                r_oofuns.append(tmp)
            r = set(r_oovars)
            r.update(*r_oofuns)
            self.dep = r    
            # /NEW
            
        return self.dep


    """                                                getFunc                                             """
    def _getFunc(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            raise FuncDesignerException('at least one argument is required')
        if (len(kwargs) == 1 and 'name' not in kwargs) or len(kwargs) > 1:
            raise FuncDesignerException('only "name" is allowed for oofun keyword arguments')
        if len(args) != 0:
            if type(args[0]) != str:
                assert not isinstance(args[0], oofun), "you can't invoke oofun on another one oofun"
            else:
                self.name = args[0]
                return self
        else:
                self.name = kwargs['name']
                return self
                
        return self._getFuncCalcEngine(*args, **kwargs)
        
        #!!!!!!!!!!!!!!!!!!! TODO: HANDLE FIXED OOFUNCS
        
#        if self.fixed and hasattr(self, 'f_key_prev'):
#            # TODO: remove duplicated code from the func
#            if isinstance(self.f_val_prev, ndarray) or isinstance(self.f_val_prev, float):
#                return atleast_1d(copy(self.f_val_prev))
#            else:
#                return deepcopy(self.f_val_prev)
    def _getFuncCalcEngine(self, *args, **kwargs):
        dep = self._getDep()
        x = args[0]
        CondSamePointByID = True if type(x) == ooPoint and self._point_id == x._id else False

        cond_same_point = CondSamePointByID or \
        (self.isCostly and 'f_key_prev' in self.__dict__  and all([array_equal(x[elem], self.f_key_prev[elem.name]) for elem in dep]))
        
        if cond_same_point:
            self.same += 1
            if isscalar(self.f_val_prev):
                return copy(self.f_val_prev)
            elif hasattr(self.f_val_prev, 'copy'):
                return self.f_val_prev.copy()
            else: 
                return deepcopy(self.f_val_prev) # Is it used somewhere?
        self.evals += 1
        
        if type(self.args) != tuple:
            self.args = (self.args, )
        Input = self._getInput(x)
        if self.args != ():
            Input += self.args
        tmp = self.fun(*Input)
        if isinstance(tmp, (list, tuple)):
            tmp = hstack(tmp)
        
        #if self._c != 0.0: tmp += self._c
        
        #self.outputTotalLength = ([asarray(elem).size for elem in self.fun(*Input)])#self.f_val_prev.size # TODO: omit reassigning
        
        if type(x) == ooPoint: 
            self._point_id = x._id
        
        if (type(x) == ooPoint or self.isCostly) and self.input[0] is not None:
            # TODO: simplify it
            self.f_val_prev = tmp
            self.f_key_prev = dict([(elem.name, copy(x[elem])) for elem in dep])
            
            if hasattr(self.f_val_prev, 'copy'): return self.f_val_prev.copy()
            elif isscalar(self.f_val_prev):
                return copy(self.f_val_prev)
            else:
                return deepcopy(self.f_val_prev) # Is it used somewhere?
        else:
            return tmp


    """                                                getFunc                                             """
    __call__ = lambda self, *args, **kwargs: self._getFunc(*args, **kwargs)


    """                                              derivatives                                           """
    def D(self, x, Vars=None, fixedVars = None, resultKeysType = 'vars', useSparse = False, exactShape = False, diffVarsID = -1):
        
        # resultKeysType doesn't matter for the case isinstance(Vars, oovar)
        if Vars is not None and fixedVars is not None:
            raise FuncDesignerException('No more than one argument from "Vars" and "fixedVars" is allowed for the function')
        assert type(Vars) != ndarray and type(fixedVars) != ndarray
        if not isinstance(x, ooPoint): x = ooPoint(x)
        #self._directlyDwasInwolved = True
        #if not hasattr(self, '_prev_D_Vars_key'):
            #sameDerivativeVariables = False
            #self._prev_D_Vars_key = None
            #self._prev_D_fixedVars_key = None
            

        #sameDerivativeVariables = True if diffVarsID == self._lastDiffVarsID or (self._prev_D_Vars_key is Vars and self._prev_D_fixedVars_key is fixedVars) else False
        #self._prev_D_Vars_key = set([Vars] if not isinstance(Vars, (list, tuple, set)) else Vars)
        #self._prev_D_fixedVars_key = set([fixedVars] if not isinstance(fixedVars, (list, tuple, set)) else fixedVars)
        
        #print 'sameDerivativeVariables:', sameDerivativeVariables
        
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
        r = self._D(x, diffVarsID, Vars, fixedVars, useSparse = useSparse)
        if isinstance(Vars, oofun):
            if Vars.is_oovar:
                return Vars(r)
            else: 
                # TODO: handle it with input of type list/tuple/etc as well
                raise FuncDesignerException('Cannot perform differentiation by non-oovar input')
        else:
            if resultKeysType == 'names':
                raise FuncDesignerException("""This possibility is out of date, 
                if it is still present somewhere in FuncDesigner doc inform developers""")
            elif resultKeysType == 'vars':
                rr = {}
                tmpDict = Vars if Vars is not None else x
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TODO: remove the cycle!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                for oov, val in x.items():
                    #if not isinstance(oov not in r, bool): print oov not in r
                    if oov not in r or (fixedVars is not None and oov in fixedVars):
                        continue
                    tmp = r[oov]
                    if useSparse == False and hasattr(tmp, 'toarray'): tmp = tmp.toarray()
                    if not exactShape and not isspmatrix(tmp):
                        if tmp.size == 1: tmp = asscalar(tmp)
                        elif min(tmp.shape) == 1: tmp = tmp.flatten()
                    rr[oov] = tmp
                return rr
            else:
                raise FuncDesignerException('Incorrect argument resultKeysType, should be "vars" or "names"')
            
            
    def _D(self, x, diffVarsID, Vars=None, fixedVars = None, useSparse = 'auto'):
        if self.is_oovar: 
            return {} if (fixedVars is not None and self in fixedVars) or (Vars is not None and self not in Vars) \
            else {self:Eye(asarray(x[self].size))}
            
        if self.input[0] is None: return {} # fixed oofun. TODO: implement input = [] properly
            
        if self.discrete: 
            return {}
            #raise FuncDesignerException('The oofun or oovar instance has been declared as discrete, no derivative is available')
        
        CondSamePointByID = True if isinstance(x, ooPoint) and self._point_id1 == x._id else False
        sameDerivativeVariables = diffVarsID == self._lastDiffVarsID 
        
        dep = self._getDep()
        ##########################
        
        # TODO: optimize it. Omit it for simple cases.
        #isTransmit = self._usedIn == 1 # Exactly 1! not 0, 2, ,3, 4, etc
        #involveStore = not isTransmit or self._directlyDwasInwolved
        involveStore = self.isCostly

        #cond_same_point = hasattr(self, 'd_key_prev') and sameDerivativeVariables and (CondSamePointByID or (involveStore and         all([array_equal(x[elem], self.d_key_prev[elem]) for elem in dep])))
        
        cond_same_point = sameDerivativeVariables and (CondSamePointByID or (involveStore and hasattr(self, 'd_key_prev') and all([array_equal(x[elem], self.d_key_prev[elem]) for elem in dep])))
        
        if cond_same_point:
            self.same_d += 1
            #return deepcopy(self.d_val_prev)
            return dict([(key, Copy(val)) for key, val in self.d_val_prev.items()])
        else:
            self.evals_d += 1

        if isinstance(x, ooPoint): self._point_id1 = x._id
        if diffVarsID != -1: self._lastDiffVarsID = diffVarsID

        derivativeSelf = self._getDerivativeSelf(x, Vars, fixedVars)

        r = Derivative()
        ac = -1
        for i, inp in enumerate(self.input):
            if not isinstance(inp, oofun): continue
            if inp.discrete: continue

            if inp.is_oovar: 
                if (Vars is not None and inp not in Vars) or (fixedVars is not None and inp in fixedVars):
                    continue                
                ac += 1
                tmp = derivativeSelf[ac]
                assert tmp.ndim > 1 

                if inp in r:
                    if prod(tmp.shape) <= prod(r[inp].shape) and type(r[inp]) == type(tmp) == ndarray: # some sparse matrices has no += implemented 
                        r[inp] += tmp
                    else:
                        r[inp] = r[inp] + tmp
                else:
                    r[inp] = tmp
            else:
                ac += 1
                
                elem_d = inp._D(x, diffVarsID, Vars=Vars, fixedVars=fixedVars, useSparse = useSparse) 
                
                t1 = derivativeSelf[ac]
                
                for key, val in elem_d.items():
                    #if isscalar(val) or val.ndim < 2: val = atleast_2d(val)
                    if isscalar(val) or isscalar(t1) or prod(t1.shape)==1 or prod(val.shape)==1:
                        rr = t1 * val
                    else:
                        if val.ndim < 2: val = atleast_2d(val)
                        if useSparse is False:
                            t2 = val
                        else:
                            t1, t2 = self._considerSparse(t1, val)
                        if t1.ndim > 1 or t2.ndim > 1:
                            # warning! t1,t2 can be sparse matrices, so I don't use t = atleast_2d(t) directly
                            if t2.ndim < 2: 
                                assert t1.ndim > 1, 'error in FuncDesigner kernel, inform developers'
                                if t1.shape[1] != t2.shape[0]:
                                    t2 = t2.reshape(1, -1)
                        else:
                            # hence these are ndarrays
                            if self._getFuncCalcEngine(x).size > 1:
                                t1 = t1.reshape(-1, 1)
                                t2 = t2.reshape(1, -1)
                            else:
                                t1 = t1.reshape(1, -1)
                                t2 = t2.reshape(-1, 1)
                        
                        if not (isinstance(t1,  ndarray) and isinstance(t2,  ndarray)):
                            # CHECKME: is it trigger somewhere?
                            if not scipyInstalled:
                                self.pWarn(scipyAbsentMsg)
                                rr = atleast_1d(dot(t1, t2))
                            else:
                                t1 = t1 if isinstance(t1, scipy.sparse.csc_matrix) else t1.tocsc() if isspmatrix(t1)  else scipy.sparse.csc_matrix(t1)
                                t2 = t2 if isinstance(t2, scipy.sparse.csr_matrix) else t2.tocsr() if isspmatrix(t2)  else scipy.sparse.csr_matrix(t2)
                                if t2.shape[0] != t1.shape[1]:
                                    if t2.shape[1] == t1.shape[1]:
                                        t2 = t2.T
                                    else:
                                        raise FuncDesignerException('incorrect shape in FuncDesigner function _D(), inform developers about the bug')
                                rr = t1._mul_sparse_matrix(t2)
                                if useSparse is False:
                                    rr = rr.toarray() 
                        else:
                            rr = atleast_1d(dot(t1, t2))
                    #assert rr.ndim>1
                        
                    if key in r:
                        if isinstance(r[key], ndarray) and hasattr(rr, 'toarray'): # i.e. rr is sparse matrix
                            rr = rr.toarray() # I guess r[key] will hardly be all-zeros
                        elif hasattr(r[key], 'toarray') and isinstance(rr, ndarray): # i.e. r[key] is sparse matrix
                            r[key] = r[key].toarray()
                        if rr.size == r[key].size and type(rr) == type(r[key]) == ndarray: 
                            r[key] += rr
                        else: 
                            r[key] = r[key] + rr
                    else:
                        r[key] = rr
        
        dp = dict([(key, Copy(value)) for key, value in r.items()])
        
        self.d_val_prev = dp
        self.d_key_prev = dict([(elem, Copy(x[elem])) for elem in dep])
        return r

    # TODO: handle 2**15 & 0.25 as parameters
    def _considerSparse(self, t1, t2):  
        if prod(t1.shape) * prod(t2.shape) > 2**15 and   (isinstance(t1, ndarray) and t1.nonzero()[0].size < 0.25*t1.size) or \
        (isinstance(t2, ndarray) and t2.nonzero()[0].size < 0.25*t2.size):
            if scipy is None: 
                self.pWarn(scipyAbsentMsg)
                return t1,  t2
            if not isinstance(t1, scipy.sparse.csc_matrix): 
                t1 = scipy.sparse.csc_matrix(t1)
            if t1.shape[1] != t2.shape[0]: # can be from flattered t1
                assert t1.shape[0] == t2.shape[0], 'bug in FuncDesigner Kernel, inform developers'
                t1 = t1.T
            if not isinstance(t2, scipy.sparse.csr_matrix): 
                t2 = scipy.sparse.csr_matrix(t2)
        return t1,  t2

    def _getDerivativeSelf(self, x, Vars,  fixedVars):
        Input = self._getInput(x)
#        if hasattr(self, 'size') and isscalar(self.size): nOutput = self.size
#        else: nOutput = self(x).size 
        hasUserSuppliedDerivative = self.d is not None
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
                        derivativeSelf.append(get_d1(self.fun, Input, diffInt=self.diffInt, stencil = self.stencil, \
                                                     args=self.args, varForDifferentiation = i, pointVal = self._getFuncCalcEngine(x), exactShape = True))
                    else:
                        # !!!!!!!!!!!!!! TODO: add check for user-supplied derivative shape
                        tmp = deriv(*Input)
                        if isscalar(tmp) or type(tmp) in (ndarray, tuple, list): # i.e. not a scipy.sparse matrix
                            tmp = atleast_2d(tmp)
                            
                            ########################################

                            Tmp = len(Input[i])
                            if tmp.shape[1] != Tmp: 
                                # TODO: add debug msg
#                                print('incorrect shape in FD AD _getDerivativeSelf')
#                                print tmp.shape[0], nOutput, tmp
                                if tmp.shape[0] != Tmp: raise FuncDesignerException('error in getDerivativeSelf()')
                                tmp = tmp.T
                                    
                            ########################################
                            
                        derivativeSelf.append(tmp)
            else:
                tmp = self.d(*Input)
                if isscalar(tmp) or type(tmp) in (ndarray, tuple, list): # i.e. not a scipy.sparse matrix
                    tmp = atleast_2d(tmp)
                    expectedTotalInputLength = sum([len(elem) for elem in Input])
                    if tmp.shape[1] != expectedTotalInputLength: 
                        # TODO: add debug msg
                        if tmp.shape[0] != expectedTotalInputLength: raise FuncDesignerException('error in getDerivativeSelf()')
                        tmp = tmp.T
                        
                ac = 0
                if isinstance(tmp, ndarray) and hasattr(tmp, 'toarray'): tmp = tmp.A # is dense matrix
                if not isinstance(tmp, ndarray):
                    csc_tmp = tmp.tocsc()
                for i, inp in enumerate(Input):
                    if self.input[i].discrete or (self.input[i].is_oovar and ((Vars is not None and self.input[i] not in Vars) or (fixedVars is not None and self.input[i] in fixedVars))):
                        ac += inp.size
                        continue                                    
                    if isinstance(tmp, ndarray):
                        TMP = tmp[:, ac:ac+inp.size]
                    else: # scipy.sparse matrix
                        TMP = csc_tmp[:, ac:ac+inp.size]
                    ac += inp.size
                    derivativeSelf.append(TMP)
                    
            # TODO: is it required?
#                if not hasattr(self, 'outputTotalLength'): self(x)
#                
#                if derivativeSelf.shape != (self.outputTotalLength, self.inputTotalLength):
#                    s = 'incorrect shape for user-supplied derivative of oofun '+self.name+': '
#                    s += '(%d, %d) expected, (%d, %d) obtained' % (self.outputTotalLength, self.inputTotalLength,  derivativeSelf.shape[0], derivativeSelf.shape[1])
#                    raise FuncDesignerException(s)
        else:
            if Vars is not None or fixedVars is not None: raise FuncDesignerException("sorry, custom oofun derivatives don't work with Vars/fixedVars arguments yet")
            if not DerApproximatorIsInstalled:
                raise FuncDesignerException('To perform this operation you should have DerApproximator installed, see http://openopt.org/DerApproximator')
                
            derivativeSelf = get_d1(self.fun, Input, diffInt=self.diffInt, stencil = self.stencil, args=self.args, pointVal = self._getFuncCalcEngine(x), exactShape = True)
            if type(derivativeSelf) == tuple:
                derivativeSelf = list(derivativeSelf)
            elif type(derivativeSelf) != list:
                derivativeSelf = [derivativeSelf]
        
        assert all([elem.ndim > 1 for elem in derivativeSelf])
        return derivativeSelf

    def D2(self, x):
        raise FuncDesignerException('2nd derivatives for obj-funcs are not implemented yet')

    def check_d1(self, point):
        if self.d is None:
            self.disp('Error: no user-provided derivative(s) for oofun ' + self.name + ' are attached')
            return # TODO: return non-void result
        separator = 75 * '*'
        self.disp(separator)
        assert type(self.d) != list
        val = self(point)
        input = self._getInput(point)
        ds= self._getDerivativeSelf(point, Vars=None,  fixedVars=None)
        self.disp(self.name + ': checking user-supplied gradient')
        self.disp('according to:')
        self.disp('    diffInt = ' + str(self.diffInt)) # TODO: ADD other parameters: allowed epsilon, maxDiffLines etc
        self.disp('    |1 - info_user/info_numerical| < maxViolation = '+ str(self.maxViolation))        
        j = -1
        for i in xrange(len(self.input)):#var in Vars:
            var = self.input[i]
            if len(self.input) > 1: self.disp('by input variable number ' + str(i) + ':')
            if isinstance(self.d, tuple) and self.d[i] is None:
                self.disp('user-provided derivative for input number ' + str(i) + ' is absent, skipping the one;')
                self.disp(separator)
                continue
            if not isinstance(self.input[i], oofun):
                self.disp('input number ' + str(i) + ' is not oofun instance, skipping the one;')
                self.disp(separator)
                continue
            j += 1
            check_d1(lambda *args: self.fun(*args), ds[j], input, \
                 func_name=self.name, diffInt=self.diffInt, pointVal = val, args=self.args, \
                 stencil = max((3, self.stencil)), maxViolation=self.maxViolation, varForCheck = i)
    
    
    # TODO: should broadcast return non-void result?

    def _broadcast(self, func, *args, **kwargs):
        if self._broadcast_id == oofun._BroadCastID: 
            return # already done for this one
            
        self._broadcast_id = oofun._BroadCastID
        
        # TODO: possibility of reverse order?
        if self.input is not None:
            for inp in self.input: 
                if not isinstance(inp, oofun): continue
                inp._broadcast(func, *args, **kwargs)
        for c in self.attachedConstraints:
            c._broadcast(func, *args, **kwargs)
        func(self)
        
        """                                             End of class oofun                                             """

# TODO: make it work for ooSystem as well
def broadcast(func, oofuncs, *args, **kwargs):
    oofun._BroadCastID += 1
    for oof in oofuncs:
        oof._broadcast(func, *args, **kwargs)

def _getAllAttachedConstraints(oofuns):
    from FuncDesigner import broadcast
    r = set()
    def F(oof):
        #print len(oof.attachedConstraints)
        r.update(oof.attachedConstraints)
    broadcast(F, oofuns)
    return r

class BooleanOOFun(oofun):
    _unnamedBooleanOOFunNumber = 0
    discrete = True
    # an oofun that returns True/False
    def __init__(self, oofun_Involved, *args, **kwargs):
        oofun.__init__(self, oofun_Involved, *args, **kwargs)
        self.input = oofun_Involved.input
        BooleanOOFun._unnamedBooleanOOFunNumber += 1
        self.name = 'unnamed_boolean_oofun_' + str(BooleanOOFun._unnamedBooleanOOFunNumber)
        
    def size(self, *args, **kwargs): raise FuncDesignerException('currently BooleanOOFun.size() is disabled')
    def D(self, *args, **kwargs): raise FuncDesignerException('currently BooleanOOFun.D() is disabled')
    def _D(self, *args, **kwargs): raise FuncDesignerException('currently BooleanOOFun._D() is disabled')

class BaseFDConstraint(BooleanOOFun):
    isConstraint = True
    tol = 0.0 
    expected_kwargs = set(['tol', 'name'])
    #def __getitem__(self, point):

    def __call__(self, *args,  **kwargs):
        expected_kwargs = self.expected_kwargs
        if not set(kwargs.keys()).issubset(expected_kwargs):
            raise FuncDesignerException('Unexpected kwargs: should be in '+str(expected_kwargs)+' got: '+str(kwargs.keys()))
            
        for elem in expected_kwargs:
            if elem in kwargs:
                setattr(self, elem, kwargs[elem])
        
        if len(args) > 1: raise FuncDesignerException('No more than single argument is expected')
        
        if len(args) == 0:
           if len(kwargs) == 0: raise FuncDesignerException('You should provide at least one argument')
           return self
        
        if isinstance(args[0], dict): # is FD Point
            val = self.oofun(args[0])
            if any(isnan(val)):
                return False
            Tol = max((0.0, self.tol))
            if any(atleast_1d(self.lb-val)>Tol):
                return False
            elif any(atleast_1d(val-self.ub)>Tol):
                return False
            return True
        elif isinstance(args[0], str):
            self.name = args[0]
            return self
        else:
            raise FuncDesignerException('unexpected type: '+type(args[0]))

    def __init__(self, oofun_Involved, *args, **kwargs):
        BooleanOOFun.__init__(self, oofun_Involved, *args, **kwargs)
        #oofun.__init__(self, lambda x: oofun_Involved(x), input = oofun_Involved)
        if len(args) != 0:
            raise FuncDesignerException('No args are allowed for FuncDesigner constraint constructor, only some kwargs')
            
        # TODO: replace self.oofun by self.engine
        self.oofun = oofun_Involved
        

class SmoothFDConstraint(BaseFDConstraint):
        
    __getitem__ = lambda self, point: self.__call__(point)
        
    def __init__(self, *args, **kwargs):
        BaseFDConstraint.__init__(self, *args, **kwargs)
        self.lb, self.ub = -inf, inf
        for key in kwargs.keys():
            if key in ['lb', 'ub']:
                setattr(self, key, asfarray(kwargs[key]))
            else:
                raise FuncDesignerException('Unexpected key in FuncDesigner constraint constructor kwargs')
    

class NonLinearConstraint(SmoothFDConstraint):
    def __init__(self, *args, **kwargs):
        SmoothFDConstraint.__init__(self, *args, **kwargs)
        
        
class BoxBoundConstraint(SmoothFDConstraint):
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

def atleast_oofun(arg):
    if isinstance(arg, oofun):
        return arg
    elif hasattr(arg, 'copy'):
        tmp = arg.copy()
        return oofun(lambda *args: tmp, is_linear=True, isConstraint = True)
    elif isscalar(arg):
        tmp = array(arg, 'float')
        return oofun(lambda *args: tmp, is_linear=True, isConstraint = True)
    else:
        return oofun(lambda *args, **kwargs: arg(*args,  **kwargs))
        #raise FuncDesignerException('incorrect type for the function _atleast_oofun')

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
