from numpy import asscalar, diag, eye, isscalar, asfarray
scipyInstalled = True
try:
    import scipy
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

def Eye(n): 
    if not scipyInstalled and n>150: 
        pWarn(scipyAbsentMsg)
    if n == 1:
        return 1.0
    elif n <= 16 or not scipyInstalled: 
        return eye(n) 
    else:  
        return scipy.sparse.identity(n) 

def Diag(x):
    if not scipyInstalled and len(x)>150: 
        pWarn(scipyAbsentMsg)
    if isscalar(x): return x
    elif len(x) == 1: return asfarray(x)
    elif len(x) < 16 or not scipyInstalled: return diag(x)
    else: return scipy.sparse.spdiags(x, [0], len(x), len(x)) 
