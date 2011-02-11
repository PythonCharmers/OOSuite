#! /usr/bin/env python

from ooVersionNumber import __version__
from oo import *

from kernel.GUI import manage
from kernel.oologfcn import OpenOptException
from kernel.nonOptMisc import oosolver

try:
    import enthought
    s = """
    Seems like you are using Enthought Python Distribution; 
    consider using free GPL-licensed alternatives 
    PythonXY (http://www.pythonxy.com) or
    Sage (http://sagemath.org) instead.
    """
    print(s)
except ImportError:
    pass
    
#__all__ = filter(lambda s:not s.startswith('_'),dir())

#from numpy.testing import NumpyTest
#test = NumpyTest().test


