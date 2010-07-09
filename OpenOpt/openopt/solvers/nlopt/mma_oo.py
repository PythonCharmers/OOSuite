from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class mma(NLOPT_BASE):
    __name__ = 'mma'
    __alg__ = "Method of Moving Asymptotes"
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'c']
    
    # TODO: implement 'A', 'Aeq', 'b', 'beq',
    #__optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    
    #properTextOutput = True
    
    #TODO: check it!
    #_canHandleScipySparse = True
    
    #__funcForIterFcnConnection__ = 'f'
    
    
    def __init__(self): pass
    def __solver__(self, p):
        # TODO: CONSTRAINTS!
        NLOPT_AUX(p, nlopt.LD_MMA)
