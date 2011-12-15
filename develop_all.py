try:
    import setuptools
except:
    print('you should have setuptools installed (http://pypi.python.org/pypi/setuptools), for some Linux distribs you can get it via [sudo] apt-get install python-setuptools')
    print('press Enter for exit...')
    raw_input()
    exit()

import os, sys
for moduleName in ['DerApproximator', 'FuncDesigner', 'OpenOpt', 'SpaceFuncs']:
    print(moduleName + ' in-place installation:')
    os.chdir(moduleName) 
    os.system('\"%s\" setup.py develop' % sys.executable)
    #os.system('%s setup.py develop' % sys.executable)
    os.chdir('..')
