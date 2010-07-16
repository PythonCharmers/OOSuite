from NLOPT_AUX import NLOPT_AUX
from NLOPT_BASE import NLOPT_BASE
import nlopt

class slmvm1(NLOPT_BASE):
    __name__ = 'slmvm1'
    __alg__ = 'Shifted limited-memory variable-metric, rank 1'
    __authors__ = 'Prof. Ladislav Luksan'
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    
    #__funcForIterFcnConnection__ = 'f'
    
    def __init__(self): pass
    def __solver__(self, p):
        # TODO: CONSTRAINTS!
        NLOPT_AUX(p, nlopt.LD_VAR1)
