from ooVar import oovar, oovars
from ooFun import _getAllAttachedConstraints, broadcast, ooarray, ooFun as oofun
from ooSystem import ooSystem as oosystem
from translator import FuncDesignerTranslator as ootranslator
from ooPoint import ooPoint as oopoint
from sle import sle
from ode import ode
from overloads import *
#from overloads import _sum as sum
from misc import FuncDesignerException, _getDiffVarsID
from interpolate import scipy_UnivariateSpline as interpolator
from integrate import integrator
__version__ = '0.33'

try:
    import enthought
    s = """
    Seems like you are using FuncDesigner from Enthought Python Distribution; 
    consider using free GPL-licensed alternatives 
    PythonXY (http://www.pythonxy.com) or
    Sage (http://sagemath.org) instead.
    """
    print(s)
except ImportError:
    pass
