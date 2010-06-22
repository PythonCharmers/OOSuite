#! /usr/bin/env python

from ooVersionNumber import __version__
from oo import *

from kernel.GUI import manage
from kernel.oologfcn import OpenOptException
from kernel.nonOptMisc import oosolver

#__all__ = filter(lambda s:not s.startswith('_'),dir())

#from numpy.testing import NumpyTest
#test = NumpyTest().test


