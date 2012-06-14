import os, sys
curr_dir = ''.join([elem + os.sep for elem in __file__.split(os.sep)[:-1]])
sys.path += [curr_dir]

__version__ = '0.38'

from ooVar import oovar, oovars
#from ooFun import _getAllAttachedConstraints, broadcast, ooFun as oofun, AND, OR, NOT, NAND, NOR, XOR
from ooFun import _getAllAttachedConstraints, broadcast, oofun, AND, OR, NOT, NAND, NOR, XOR

from ooSystem import ooSystem as oosystem
from translator import FuncDesignerTranslator as ootranslator

from ooPoint import ooPoint as oopoint, ooMultiPoint 
#from logic import *
from stochastic import discrete
from ooarray import ooarray

def IMPLICATION(condition, *args):
    if len(args) == 1 and isinstance(args[0], (tuple, set, list, ndarray)):
        return ooarray([IMPLICATION(condition, elem) for elem in args[0]])
    elif len(args) > 1:
        return ooarray([IMPLICATION(condition, elem) for elem in args])
    return NOT(condition & NOT(args[0]))
    
ifThen = IMPLICATION

from sle import sle
from ode import ode
from dae import dae
from overloads import *
from stencils import d, d2
#from overloads import _sum as sum
from FDmisc import FuncDesignerException, _getDiffVarsID
from interpolate import scipy_UnivariateSpline as interpolator
from integrate import integrator


isE = False
try:
    import enthought
    isE = True
except ImportError:
    pass
try:
    import envisage
    import mayavi
    isE = True
except ImportError:
    pass
try:
    import xy
    isE = False
except ImportError:
    pass
  
if isE:
    s = """
    Seems like you are using OpenOpt from 
    commercial Enthought Python Distribution;
    consider using free GPL-licensed alternatives
    PythonXY (http://www.pythonxy.com) or
    Sage (http://sagemath.org) instead.
    """
    print(s)
    
del(isE, curr_dir, os, sys)
