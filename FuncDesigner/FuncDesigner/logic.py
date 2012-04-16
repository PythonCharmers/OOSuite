from FDmisc import FuncDesignerException
from .ooFun import oofun, BooleanOOFun, nlh_and, nlh_not
import numpy as np

def AND(*args):
    Args = args[0] if len(args) == 1 and type(args[0]) in (tuple, list, set) else args
    assert not isinstance(args[0], np.ndarray), 'unimplemented yet' 
    for arg in Args:
        if not isinstance(arg, oofun):
            raise FuncDesignerException('FuncDesigner logical AND currently is implemented for oofun instances only')
    #if other is True: return self
    
    
    r = BooleanOOFun(np.logical_and, Args, vectorized = True)
    r.nlh = lambda *arguments: nlh_and(Args, r._getDep(), *arguments)
    r.oofun = r
    return r

#def NOT(Args):
def NOT(_bool_oofun):
    assert not isinstance(_bool_oofun, (np.ndarray, list, tuple, set)), 'unimplemented yet' 
    #Args = args[0] if len(args) == 1 and type(args[0]) in (tuple, list, set) else args
    #Args = args if type(args) in (tuple, list, set) else [args]
    if not isinstance(_bool_oofun, oofun):
        raise FuncDesignerException('FuncDesigner logical AND currently is implemented for oofun instances only')
#    for arg in Args:
#        if not isinstance(arg, oofun):
#            raise FuncDesignerException('FuncDesigner logical AND currently is implemented for oofun instances only')
#            
    #if other is True: return False
    
    
    r = BooleanOOFun(np.logical_not, [_bool_oofun], vectorized = True)
    r.nlh = lambda *arguments: nlh_not(_bool_oofun, r._getDep(), *arguments)
    r.oofun = r
    return r

def OR(*args):
    Args = args[0] if len(args) == 1 and type(args[0]) in (tuple, list, set) else args
    assert not isinstance(args[0], np.ndarray), 'unimplemented yet' 
    for arg in Args:
        if not isinstance(arg, oofun):
            raise FuncDesignerException('FuncDesigner logical AND currently is implemented for oofun instances only')
    
    r = ~ AND([~elem for elem in Args])
    #r.fun = np.logical_or
    r.oofun = r
    return r
    
    
    
    
    
