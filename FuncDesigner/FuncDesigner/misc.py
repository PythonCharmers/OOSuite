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
    __array_priority__ = 15# set it greater than 1 to prevent invoking numpy array __mul__ etc
    
    def __init__(self, arr, scalarMultiplier=1.0):
        assert arr.ndim <= 1
        self.diag = arr#.copy() 
        self.scalarMultiplier = scalarMultiplier
        self.size = arr.size
        
    
    def toarray(self):
        return np.diag(self.diag * self.scalarMultiplier)
    
    def resolve(self, useSparse):
        if useSparse in (True, 'auto') and scipyInstalled and self.size > 50:
            r = SP.lil_matrix((self.size, self.size))
            r.setdiag(self.diag*self.scalarMultiplier)
            return 
        else: 
            return self.toarray()
#        elif useSparse == False or not scipyInstalled:
#            return self.toarray()
#        else:
#            assert 0, 'error in FD kernel'
    
    def __mul__(self, item):
        r = diagonal(self.diag, self.scalarMultiplier) # mb use copy instead?
        if np.isscalar(item):
            r.scalarMultiplier *= item
        elif type(item) == DiagonalType:#diagonal:
            r.scalarMultiplier *= item.scalarMultiplier
            
            ##################
            # Not in-place modification!
            r.diag = r.diag * item.diag
            ##################
        elif type(item) == np.ndarray:
            if item.size == 1:
                r.scalarMultiplier *= np.asscalar(item)
            else:
                r = np.dot(self.resolve(False), item)
        else:
            assert SP.isspmatrix(item)
            if prod(item.shape) == 1:
                r.scalarMultiplier *= item[0, 0]
            else:
                tmp = SP.lil_matrix(r.resolve(True)) # r.resolve(True) can yield dense ndarray
                r = tmp._mul_sparse_matrix(item)
        return r
    
    def __getattr__(self, attr):
        if attr == 'T': return self
        elif attr == 'shape': return self.size, self.size
        raise AttributeError('you are trying to obtain incorrect attribute "%s" for FuncDesigner diagonal' %attr)
    
    def __rmul__(self, item):
        return self.__mul__(item) if isscalar(item) else self.__mul__(item.T).T

DiagonalType = type(diagonal(np.array(0)))

Eye = lambda n: 1.0 if n == 1 else diagonal(np.ones(n))

def Diag(x):
    if isscalar(x): return x
    else: return diagonal(x)
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
