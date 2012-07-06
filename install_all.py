try:
    import setuptools
except:
    print('you should have setuptools installed (http://pypi.python.org/pypi/setuptools), for some Linux distribs you can get it via [sudo] apt-get install python-setuptools')
    print('press Enter for exit...')
    raw_input()
    exit()

import os, sys
(filepath, filename) = os.path.split(__file__)

for moduleName in ['DerApproximator', 'FuncDesigner', 'OpenOpt', 'SpaceFuncs']:
    print(moduleName + ' installation:')
    os.chdir(((filepath + os.sep) if filepath != '' else '') + moduleName)
    os.system('\"%s\" setup.py install' % sys.executable)
    #os.system('%s setup.py install' % sys.executable)
    os.chdir('..')

