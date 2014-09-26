"""
A setup script for all OpenOpt-related packages. Allows all four to be
installable easily with ``pip``.
"""

import os, sys
(filepath, _) = os.path.split(__file__)

for moduleName in ['DerApproximator', 'FuncDesigner', 'OpenOpt', 'SpaceFuncs']:
    print(moduleName + ' installation:')
    os.chdir(((filepath + os.sep) if filepath != '' else '') + moduleName)
    os.system('\"%s\" setup.py install' % sys.executable)
    #os.system('%s setup.py install' % sys.executable)
    os.chdir('..')

