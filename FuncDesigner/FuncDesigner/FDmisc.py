from numpy import asscalar, isscalar, asfarray, ndarray, prod
import numpy as np
from baseClasses import MultiArray

scipyInstalled = True
try:
    import scipy
    import scipy.sparse as SP
except:
    scipyInstalled = False
    
from baseClasses import Stochastic

class FuncDesignerException(BaseException):
    def __init__(self,  msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def checkSizes(a, b):
    if a.size != 1 and b.size != 1 and a.size != b.size:
        raise FuncDesignerException('operation of oovar/oofun ' + a.name + \
        ' and object with inappropriate size:' + str(a.size) + ' vs ' + b.size)

scipyAbsentMsg = 'Probably scipy installation could speed up running the code involved'

pwSet = set()
def pWarn(msg):
    if msg in pwSet: return
    pwSet.add(msg)
    print('FuncDesigner warning: ' + msg)


class diagonal:
    isOnes = False
    __array_priority__ = 150000# set it greater than 1 to prevent invoking numpy array __mul__ etc
    
    def __init__(self, arr, scalarMultiplier=1.0, size=0):
        #assert arr is None or arr.ndim <= 1
        self.diag = arr.copy() if arr is not None else None # may be None, then n has to be provided
        self.scalarMultiplier = scalarMultiplier if isscalar(scalarMultiplier) \
        else asscalar(scalarMultiplier) if type(scalarMultiplier) == ndarray\
        else scalarMultiplier[0, 0] if scipyInstalled and SP.isspmatrix(scalarMultiplier)\
        else raise_except()
        self.size = arr.size if size == 0 else size
        if arr is None:
            self.isOnes = True
        
    copy = lambda self: diagonal(self.diag, scalarMultiplier = self.scalarMultiplier, size = self.size)
    
    def toarray(self):
        if self.isOnes:
            tmp = np.empty(self.size)
            
            # for PyPy compatibility
            scalarMultiplier = asscalar(self.scalarMultiplier) if type(self.scalarMultiplier) == ndarray else self.scalarMultiplier
            
            tmp.fill(scalarMultiplier)
            return np.diag(tmp)
        else:
            return np.diag(self.diag * self.scalarMultiplier)
    
    def resolve(self, useSparse):
        if useSparse in (True, 'auto') and scipyInstalled and self.size > 50:
            if self.isOnes:
                tmp = np.empty(self.size)
                tmp.fill(self.scalarMultiplier)
            else:
                tmp = self.diag*self.scalarMultiplier
            return SP.dia_matrix((tmp,0), shape=(self.size,self.size)) 
        else:
            return self.toarray()

    def __add__(self, item):
        if type(item) == DiagonalType:
            # TODO: mb use other.diag.copy(), self.diag.copy() for more safety, especially for parallel computations?
            if self.isOnes and item.isOnes:
                return diagonal(None, self.scalarMultiplier + item.scalarMultiplier, size=self.size)
            else:
                if self.isOnes:
                    d1 = np.empty(self.size) 
                    d1.fill(self.scalarMultiplier )
                else:
                    d1 = self.diag
                if item.isOnes:
                    d2 = np.empty(item.size) 
                    d2.fill(item.scalarMultiplier )
                else:
                    d2 = item.diag
                return diagonal(d1 * self.scalarMultiplier + d2 * item.scalarMultiplier)
        elif np.isscalar(item) or type(item) == np.ndarray:
            return self.resolve(False)+item
        else: # sparse matrix
            assert SP.isspmatrix(item)
            return self.resolve(True)+item
    
    def __radd__(self, item):
        return self.__add__(item)
    
    def __neg__(self):
        return diagonal(self.diag, -self.scalarMultiplier, size=self.size)
    
    def __mul__(self, item): 
        #!!! PERFORMS MATRIX MULTIPLICATION!!!
        if np.isscalar(item):
            return diagonal(self.diag, item*self.scalarMultiplier, size=self.size)
        if type(item) == DiagonalType:#diagonal:
            scalarMultiplier = item.scalarMultiplier * self.scalarMultiplier
            if self.isOnes:
                diag = item.diag
            elif item.isOnes:
                diag = self.diag
            else:
                diag = self.diag * item.diag
            return diagonal(diag, scalarMultiplier, size=self.size) 
        elif isinstance(item, np.ndarray):
            if item.size == 1:
                return diagonal(self.diag, scalarMultiplier = np.asscalar(item)*self.scalarMultiplier, size=self.size)
            elif min(item.shape) == 1:
                #TODO: assert item.ndim <= 2 
                r = self.scalarMultiplier*item.flatten()
                if self.diag is not None: r *= self.diag
                return r.reshape(item.shape)
            else:
                # new; TODO: improve it
                if self.isOnes:
                    D = np.empty(self.size)
                    D.fill(self.scalarMultiplier)
                else:
                    D = self.scalarMultiplier * self.diag if self.scalarMultiplier != 1.0 else self.diag
                return D.reshape(-1, 1) * item # ! different shapes !
                
                
#                    T = np.dot(self.resolve(False), item)
#                    from numpy import array_equal, all
#                    assert array_equal(T.shape,  T2.shape) and all(T==T2)
#                    print '!'
                #prev
                # !!!!!!!!!! TODO:  rework it!!!!!!!!!!!
#                if self.size < 100 or not scipyInstalled:
#                    return np.dot(self.resolve(False), item)
#                else:
#                    return self.resolve(True)._mul_sparse_matrix(item)
        else:
            #assert SP.isspmatrix(item)
            if prod(item.shape) == 1:
                return diagonal(self.diag, scalarMultiplier = self.scalarMultiplier*item[0, 0], size=self.size)
            else:
                tmp = self.resolve(True)
                if not SP.isspmatrix(tmp): # currently lil_matrix and K^ works very slow on sparse matrices
                    tmp = SP.lil_matrix(tmp) # r.resolve(True) can yield dense ndarray
                return tmp._mul_sparse_matrix(item)
        #return r
    
    def __getattr__(self, attr):
        if attr == 'T': return self # TODO: mb using copy will be more safe
        elif attr == 'shape': return self.size, self.size
        elif attr == 'ndim': return 2
        raise AttributeError('you are trying to obtain incorrect attribute "%s" for FuncDesigner diagonal' %attr)
    
    def __rmul__(self, item):
        return self.__mul__(item) if isscalar(item) else self.__mul__(item.T).T
    
    def __div__(self, other):
        #TODO: check it
        if isinstance(other, np.ndarray) and other.size == 1: other = np.asscalar(other)
        if np.isscalar(other) or prod(other.shape)==1: 
            return diagonal(self.diag, self.scalarMultiplier/other, size=self.size) 
        else: 
            # TODO: check it
            return diagonal(self.diag/other if self.diag is not None else 1.0/other, self.scalarMultiplier, size=self.size) 

DiagonalType = type(diagonal(np.array([0, 0])))

Eye = lambda n: 1.0 if n == 1 else diagonal(None, size=n)

def Diag(x, *args, **kw):
    if isscalar(x) or (type(x)==ndarray and x.size == 1) or isinstance(x, (Stochastic, MultiArray)): 
        return x
    else: 
        return diagonal(asfarray(x) if x is not None else x, *args,  **kw)

class fixedVarsScheduleID:
    fixedVarsScheduleID = 0
    def _getDiffVarsID(*args):
        fixedVarsScheduleID.fixedVarsScheduleID += 1
        return fixedVarsScheduleID.fixedVarsScheduleID
DiffVarsID = fixedVarsScheduleID()
_getDiffVarsID = lambda *args: DiffVarsID._getDiffVarsID(*args)

try:
    import numpypy
    isPyPy = True
except ImportError:
    isPyPy = False

def raise_except(*args, **kwargs):
    raise FuncDesignerException('bug in FuncDesigner engine, inform developers')
    
class Extras:
    pass

