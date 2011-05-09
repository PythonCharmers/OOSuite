#! /usr/bin/env python

#from .ooVersionNumber import __version__

import os, sys
curr_dir = ''.join([elem + os.sep for elem in __file__.split(os.sep)[:-1]])
sys.path += [curr_dir, curr_dir + 'kernel']

from ooVersionNumber import __version__
from oo import *

#from kernel.GUI import manage
#from kernel.oologfcn import OpenOptException
#from kernel.nonOptMisc import oosolver

from GUI import manage
from oologfcn import OpenOptException
from nonOptMisc import oosolver

try:
    import enthought
    s = """
    Seems like you are using OpenOpt from Enthought Python Distribution; 
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


