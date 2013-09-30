import numpy as np # used in MATLAB execute
from openopt.kernel.baseSolver import *
try:
    from scipy.sparse import find, isspmatrix
except ImportError:
    isspmatrix = lambda *args, **kw: False
#from openopt.kernel.setDefaultIterFuncs import *
import os
from wh import wh
ArrAttribs = ('x0', 'lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'nc', 'nh')
FuncAttribs = ('f', 'df', 'c', 'dc', 'h', 'dh')

def FuncWrapper(p, attr):
    Func = getattr(p, attr)
    func = lambda x: Func(x.flatten())
#    r = func(x.flatten())
#    print('attr:', attr)
#    print ('r:', r)
    return func

class fmincon(baseSolver):
    
    #######################
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: patterns!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #######################
    
    __name__ = 'fmincon'
    __license__ = "proprietary"
    __authors__ = ""
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    iterfcnConnected = True
    _canHandleScipySparse = True
    #matlabExecutable = '/usr/local/MATLAB/R2012a/bin/matlab'
    matlab = 'matlab'

    def __init__(self): 
        pass
    def __solver__(self, p):
        # TODO: cons patterns
        
#        if p.nc != 0: r.append(p._getPattern(p.user.c))
#        if p.nh != 0: r.append(p._getPattern(p.user.h))
        
        Data = {'p': p, 'TolFun': p.ftol, \
        'TolCon': p.contol, 'TolX': p.xtol}
        for attr in ArrAttribs:
            Data[attr] = getattr(p, attr)
        for attr in FuncAttribs:
            Data[attr] = FuncWrapper(p, attr)#getattr(p, attr)(x.flatten())
            
        #os.system('/usr/local/MATLAB/R2012a/bin/matlab ~/Documents/MATLAB/open_wormhole.m')
        
        wh(Data, self.matlab)
        if p.istop == 0:
            p.istop = 1000
        #execfile('/home/dmitrey/Install/Wormhole-0.1a/open_wormhole.py')
        
        
        
#        exec(open('/home/dmitrey/Install/Wormhole-0.1a/open_wormhole.py').read())
#        p.xf = xf
#        import open_wormhole 
    
