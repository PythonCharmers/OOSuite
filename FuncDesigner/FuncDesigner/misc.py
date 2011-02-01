from numpy import asscalar, diag, eye, isscalar, asfarray
import numpy as np

scipyInstalled = True
try:
    import scipy
    import scipy.sparse as SP
except:
    scipyInstalled = False


class FuncDesignerException:
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
    isOnes = True
    __array_priority__ = 150000# set it greater than 1 to prevent invoking numpy array __mul__ etc
    
    def __init__(self, arr, scalarMultiplier=1.0, isOnes = False, Copy=True):
        #assert arr.ndim <= 1
        self.diag = arr.copy() if Copy else arr
        self.scalarMultiplier = scalarMultiplier
        self.size = arr.size
        self.isOnes = isOnes
        
    
    def toarray(self):
        if self.isOnes:
            tmp = np.empty(self.size)
            tmp.fill(self.scalarMultiplier)
            #if tmp.size > 1000: raise 0
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
            #raise 0
            return SP.dia_matrix((tmp,0), shape=(self.size,self.size)) 
        else:
            return self.toarray()



    
    def __add__(self, item):
        if type(item) == DiagonalType:
            # TODO: mb use other.diag.copy(), self.diag.copy() for more safety, especially for parallel computations?
            if self.isOnes and item.isOnes:
                return diagonal(self.diag, self.scalarMultiplier + item.scalarMultiplier, isOnes = True)
            else:
                return diagonal(self.diag * self.scalarMultiplier + item.diag*item.scalarMultiplier)
        elif np.isscalar(item) or type(item) == np.ndarray:
            return self.resolve(False)+item
        else: # sparse matrix
            assert SP.isspmatrix(item)
            return self.resolve(True)+item
    
    def __radd__(self, item):
        return self.__add__(item)
    
    def __neg__(self):
        return diagonal(self.diag, -self.scalarMultiplier, isOnes = self.isOnes, Copy=False)
    
    def __mul__(self, item): 
        #!!! PERFORMS MATRIX MULTIPLICATION!!!
        if np.isscalar(item):
            return diagonal(self.diag, item*self.scalarMultiplier)
        if type(item) == DiagonalType:#diagonal:
            scalarMultiplier = item.scalarMultiplier * self.scalarMultiplier
            if self.isOnes:
                diag = item.diag
            elif item.isOnes:
                diag = self.diag
            else:
                diag = self.diag * item.diag
            return diagonal(diag, scalarMultiplier, isOnes = item.isOnes and self.isOnes) 
        elif isinstance(item, np.ndarray):
            if item.size == 1:
                return diagonal(self.diag, scalarMultiplier = np.asscalar(item)*self.scalarMultiplier, isOnes = self.isOnes)
            elif min(item.shape) == 1:
                #TODO: assert item.ndim <= 2 
                return (self.scalarMultiplier*self.diag*item.flatten()).reshape(item.shape)
            else:
                # !!!!!!!!!! TODO:  rework it!!!!!!!!!!!
                if self.size < 100 or not scipyInstalled:
                    return np.dot(self.resolve(False), item)
                else:
                    return self.resolve(True)._mul_sparse_matrix(item)
        else:
            #assert SP.isspmatrix(item)
            if np.prod(item.shape) == 1:
                return diagonal(self.diag, scalarMultiplier = self.scalarMultiplier*item[0, 0])
            else:
                tmp = self.resolve(True)
                if not SP.isspmatrix(tmp): # currently lil_matrix and K^ works very slow on sparse matrices
                    tmp = SP.lil_matrix(tmp) # r.resolve(True) can yield dense ndarray
                return tmp._mul_sparse_matrix(item)
        #return r
    
    def __getattr__(self, attr):
        if attr == 'T': return self
        elif attr == 'shape': return self.size, self.size
        raise AttributeError('you are trying to obtain incorrect attribute "%s" for FuncDesigner diagonal' %attr)
    
    def __rmul__(self, item):
        return self.__mul__(item) if isscalar(item) else self.__mul__(item.T).T

DiagonalType = type(diagonal(np.array(0)))

Eye = lambda n: 1.0 if n == 1 else diagonal(np.ones(n), isOnes = True, Copy = False)

def Diag(x):
    if isscalar(x): return x
    else: return diagonal(asfarray(x), Copy = False)
#    elif len(x) == 1: return asfarray(x)
#    elif len(x) < 16 or not scipyInstalled: return diag(x)
#    else: return scipy.sparse.spdiags(x, [0], len(x), len(x)) 

#def Eye(n): 
#    if not scipyInstalled and n>150: 
#        pWarn(scipyAbsentMsg)
#    if n == 1:
#        return 1.0
#    elif n <= 16 or not scipyInstalled: 
#        return eye(n) 
#    else:  
#        return scipy.sparse.identity(n) 

#def Diag(x):
#    if not scipyInstalled and len(x)>150: 
#        pWarn(scipyAbsentMsg)
#    if isscalar(x): return x
#    elif len(x) == 1: return asfarray(x)
#    elif len(x) < 16 or not scipyInstalled: return diag(x)
#    else: return scipy.sparse.spdiags(x, [0], len(x), len(x)) 


class fixedVarsScheduleID:
    fixedVarsScheduleID = 0
    def _getDiffVarsID(*args):
        fixedVarsScheduleID.fixedVarsScheduleID += 1
        return fixedVarsScheduleID.fixedVarsScheduleID
DiffVarsID = fixedVarsScheduleID()
_getDiffVarsID = lambda *args: DiffVarsID._getDiffVarsID(*args)

def raise_except(*args, **kwargs):
    raise FuncDesignerException('bug in FuncDesigner engine, inform developers')
    
class Extras:
    pass
