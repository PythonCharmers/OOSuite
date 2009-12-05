from ooVar import oovar, oovars
from ooFun import NonLinearConstraint, oofun as OOFun
from sle import sle
from overloads import *
from misc import FuncDesignerException
import integrate, interpolate
__version__ = '0.15'

def oofun(*args, **kwargs):
    r = OOFun(*args, **kwargs)
    r.isCostly = True
    return r
