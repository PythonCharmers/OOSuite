import os, sys
for moduleName in ['DerApproximator', 'FuncDesigner', 'OpenOpt']:
    print(moduleName + ' installation:')
    os.chdir(moduleName) 
    os.system('%s setup.py develop' % sys.executable)
    os.chdir('..')
